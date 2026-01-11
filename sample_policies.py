import os

def create_sample_policies():
    """Create sample policy documents for testing."""
    os.makedirs("./policy_documents", exist_ok=True)
    
    # Data Governance Policy
    data_policy = """
DATA GOVERNANCE POLICY
Version 2.1 - Effective January 2024

Section 1: Purpose
This policy establishes guidelines for data handling, storage, and processing.

Section 4: Data Storage Requirements
4.1 All restricted and confidential data must be stored on organization-controlled infrastructure.
4.2 External data storage requires written approval from the Data Protection Officer.
4.3 Cloud storage providers must be approved vendors with SOC 2 or ISO 27001.
4.4 Data encryption is mandatory:
    - At rest: AES-256 encryption minimum
    - In transit: TLS 1.2 or higher

Section 5: Third-Party Data Processing
5.1 External data processors must undergo security assessment before engagement.
5.2 Data Processing Agreements (DPAs) are required for all third-party processors.
5.3 Regular audits of third-party processors must be conducted annually.

Section 6: User Consent
6.1 Explicit consent required for processing personal data.
6.2 Users must be informed of data storage locations and processing purposes.
6.3 Consent must be documented and auditable.
"""
    
    security_policy = """
INFORMATION SECURITY POLICY
Version 3.0 - Effective January 2024

Section 2: Access Management
2.1 Multi-factor authentication (MFA) required for all external access.
2.2 Password requirements: minimum 12 characters, rotation every 90 days.

Section 3: Data Protection
3.1 Data at rest encryption is mandatory for all sensitive data.
3.2 Backup encryption using organization-approved methods only.
3.3 Data must be anonymized or pseudonymized when possible for analytics.

Section 5: Vendor Management
5.1 All vendors handling data must complete security questionnaires.
5.2 Penetration testing results required for high-risk vendors.
"""
    
    privacy_policy = """
PRIVACY AND DATA PROTECTION POLICY
Version 1.5 - Effective January 2024

Section 3: Analytics and Tracking
3.1 Analytics tools must comply with privacy regulations.
3.2 IP address anonymization required for web analytics.
3.3 Cookie consent required before placing non-essential cookies.
3.4 Third-party analytics require data processing agreements.

Section 4: International Data Transfers
4.1 Transfers outside authorized regions require specific safeguards.
4.2 Standard Contractual Clauses (SCCs) required for EU data transfers.
"""
    
    with open("./policy_documents/data_governance_policy.txt", "w") as f:
        f.write(data_policy)
    
    with open("./policy_documents/security_policy.txt", "w") as f:
        f.write(security_policy)
    
    with open("./policy_documents/privacy_policy.txt", "w") as f:
        f.write(privacy_policy)

if __name__ == "__main__":
    create_sample_policies()