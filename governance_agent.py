from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
import json
import os
import re

class GovernanceDecision(BaseModel):
    decision: Literal["Allowed", "Not Allowed", "Conditional"]
    reason: str = Field(description="Clear explanation of the decision")
    suggested_changes: List[str] = Field(description="Changes to make action compliant")
    references: List[str] = Field(description="Policy sections that informed decision")
    risk_level: Literal["Low", "Medium", "High", "Critical"]
    alternative_actions: List[str] = Field(description="Compliant alternatives")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in decision")

class GovernanceAgent:
    def __init__(self):
        """
        Initialize with Hugging Face Inference API.
        Using Qwen2.5-72B - currently free and actively supported.
        """
        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_token:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
        
        self.client = InferenceClient(token=hf_token)
        
        # Working models as of January 2025 (free tier):
        self.model = "openai/gpt-oss-20b"
        # Alternatives that also work:
        # "mistralai/Mistral-7B-Instruct-v0.3"
        # "meta-llama/Meta-Llama-3-8B-Instruct"
    
    def create_decision_prompt(self, action: str, context: str, additional_context: str = "") -> str:
        prompt = f"""You are a Governance Agent evaluating compliance with organizational policies.

PROPOSED ACTION:
{action}

ORGANIZATIONAL CONTEXT:
{additional_context}

{context}

CRITICAL: You MUST respond with ONLY a valid JSON object containing ALL of these EXACT fields (no extra fields, no missing fields):

{{
  "decision": "Allowed" | "Not Allowed" | "Conditional",
  "reason": "detailed explanation citing specific policy sections",
  "suggested_changes": ["specific change 1", "specific change 2"],
  "references": ["Policy Section X.Y", "Policy Section Z"],
  "risk_level": "Low" | "Medium" | "High" | "Critical",
  "alternative_actions": ["alternative 1", "alternative 2"],
  "confidence_score": 0.85
}}

ALL FIELDS ARE REQUIRED:
- decision: Must be exactly "Allowed", "Not Allowed", or "Conditional"
- reason: String explaining the decision
- suggested_changes: Array of strings (can be empty array if no changes needed)
- references: Array of strings with policy section references
- risk_level: Must be exactly "Low", "Medium", "High", or "Critical"
- alternative_actions: Array of strings (can be empty array if none)
- confidence_score: Number between 0.0 and 1.0

Output ONLY the JSON object with all 7 fields. No markdown, no code blocks, no explanations."""
        return prompt
    
    def evaluate_action(self, action: str, policy_context: str, additional_context: str = "") -> GovernanceDecision:
        prompt = self.create_decision_prompt(action, policy_context, additional_context)
        
        try:
            # Use chat completion
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Initialize response_text
            response_text = None
            
            # Try text generation first (more reliable for this model)
            try:
                full_prompt = f"{prompt}"
                response_text = self.client.text_generation(
                    prompt=full_prompt,
                    model=self.model,
                    max_new_tokens=1500,
                    temperature=0.1,
                    return_full_text=False
                )
                if response_text:
                    response_text = str(response_text).strip()
                else:
                    raise ValueError("Empty response from text_generation")
                    
            except Exception as text_error:
                # Fallback to chat_completion if text_generation doesn't work
                try:
                    response = self.client.chat_completion(
                        messages=messages,
                        model=self.model,
                        max_tokens=1500,
                        temperature=0.1
                    )
                    
                    # Extract response from chat completion format
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        if hasattr(response.choices[0], 'message'):
                            if hasattr(response.choices[0].message, 'content'):
                                response_text = response.choices[0].message.content
                            else:
                                response_text = str(response.choices[0].message)
                        elif hasattr(response.choices[0], 'text'):
                            response_text = response.choices[0].text
                        else:
                            response_text = str(response.choices[0])
                    elif hasattr(response, 'generated_text'):
                        response_text = response.generated_text
                    elif isinstance(response, str):
                        response_text = response
                    else:
                        response_text = str(response)
                    
                    if response_text:
                        response_text = str(response_text).strip()
                    else:
                        raise ValueError(f"Empty response from chat_completion. Text error was: {type(text_error).__name__}: {str(text_error)}")
                        
                except Exception as chat_error:
                    raise Exception(f"Both text_generation and chat_completion failed. Text error: {type(text_error).__name__}: {str(text_error)}. Chat error: {type(chat_error).__name__}: {str(chat_error)}")
            
            # Check if response is empty or None
            if not response_text or len(str(response_text).strip()) == 0:
                raise ValueError("Empty response from API after all attempts")
            
            # Clean markdown code blocks
            response_text = str(response_text).strip()
            if len(response_text) == 0:
                raise ValueError("Empty response after cleaning")
            if response_text.startswith("```json"):
                response_text = response_text[7:].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:].strip()
            
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            
            # Find JSON in response (in case model adds extra text)
            # Try to find the JSON object by counting braces for proper nesting
            json_start = response_text.find('{')
            if json_start == -1:
                # Try to parse the entire response as-is
                json_str = response_text
            else:
                # Count braces to find the matching closing brace
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if brace_count != 0:
                    # Braces don't match, try simpler extraction
                    json_end = response_text.rfind('}') + 1
                    if json_end <= json_start:
                        json_end = len(response_text)
                
                json_str = response_text[json_start:json_end]
            
            # Try to parse JSON, with repair attempts for common issues
            decision_json = None
            json_attempts = [json_str]
            
            # Try to fix common JSON issues
            if json_str.count('"') % 2 != 0:
                # Missing quote, try adding one at the end
                json_attempts.append(json_str + '"')
            
            # Remove trailing commas before closing braces/brackets
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
            if fixed_json != json_str:
                json_attempts.append(fixed_json)
            
            # Try parsing with each attempt
            for attempt in json_attempts:
                try:
                    decision_json = json.loads(attempt)
                    break
                except json.JSONDecodeError:
                    continue
            
            if decision_json is None:
                # All parsing attempts failed - raise a proper JSONDecodeError
                raise json.JSONDecodeError("Could not parse JSON after multiple repair attempts", json_str, len(json_str))
            
            # Validate with Pydantic
            decision = GovernanceDecision(**decision_json)
            return decision
            
        except json.JSONDecodeError:
            return GovernanceDecision(
                decision="Not Allowed",
                reason="Unable to parse governance decision. The AI response was not in valid JSON format. Manual review required.",
                suggested_changes=["Review action manually with compliance team"],
                references=["Manual Review Required"],
                risk_level="High",
                alternative_actions=["Consult with compliance officer"],
                confidence_score=0.0
            )
            
        except ValidationError as e:
            # Extract what we can from the parsed JSON (validation failed, but JSON was valid)
            decision_json_dict = decision_json if 'decision_json' in locals() else {}
            
            # Try to build a valid decision with defaults for missing fields
            try:
                # Provide defaults for missing required fields
                decision_data = {
                    "decision": decision_json_dict.get("decision", "Not Allowed"),
                    "reason": decision_json_dict.get("reason", f"Validation error: Missing required fields. Original: {str(e)[:200]}"),
                    "suggested_changes": decision_json_dict.get("suggested_changes", ["Review action manually with compliance team"]),
                    "references": decision_json_dict.get("references", ["Manual Review Required"]),
                    "risk_level": decision_json_dict.get("risk_level", "High"),
                    "alternative_actions": decision_json_dict.get("alternative_actions", ["Consult with compliance officer"]),
                    "confidence_score": decision_json_dict.get("confidence_score", 0.0)
                }
                
                # Validate with corrected data
                decision = GovernanceDecision(**decision_data)
                return decision
            except Exception:
                # If even defaults fail, return hardcoded fallback
                return GovernanceDecision(
                    decision="Not Allowed",
                    reason=f"Validation error: The model response didn't match the expected format. Details: {str(e)}. Manual review required.",
                    suggested_changes=["Review action manually with compliance team", "Check that all required fields are present"],
                    references=["Manual Review Required"],
                    risk_level="High",
                    alternative_actions=["Consult with compliance officer"],
                    confidence_score=0.0
                )
            
        except Exception as e:
            return GovernanceDecision(
                decision="Not Allowed",
                reason=f"System error during evaluation: {str(e)}. Please retry or contact system administrator.",
                suggested_changes=["Retry evaluation", "Check API connectivity"],
                references=["System Error"],
                risk_level="High",
                alternative_actions=["Contact system administrator", "Try again in a few minutes"],
                confidence_score=0.0
            )