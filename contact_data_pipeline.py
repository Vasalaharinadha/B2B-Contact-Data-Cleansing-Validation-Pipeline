"""
B2B Contact Data Cleansing & Validation Pipeline
-------------------------------------------------
A comprehensive data engineering pipeline for cleaning, validating, and standardizing
B2B contact information including emails, phone numbers, company data, and job titles.

Author: Your Name
Purpose: Portfolio project demonstrating data engineering skills for B2B Sales Intelligence
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class B2BContactDataPipeline:
    """
    End-to-end data pipeline for B2B contact data processing.
    Handles cleaning, validation, standardization, and enrichment.
    """
    
    def __init__(self):
        self.stats = {
            'total_records': 0,
            'cleaned_records': 0,
            'invalid_records': 0,
            'duplicate_records': 0
        }
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw B2B contact data from various file formats.
        
        Args:
            filepath: Path to the data file (CSV, Excel, JSON)
            
        Returns:
            DataFrame containing raw contact data
        """
        logger.info(f"Loading data from {filepath}")
        
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            elif filepath.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                raise ValueError("Unsupported file format")
            
            self.stats['total_records'] = len(df)
            logger.info(f"Loaded {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_email_addresses(self, df: pd.DataFrame, email_col: str = 'email') -> pd.DataFrame:
        """
        Clean and validate email addresses.
        
        Args:
            df: Input DataFrame
            email_col: Name of email column
            
        Returns:
            DataFrame with cleaned email addresses and validation flag
        """
        logger.info("Cleaning email addresses...")
        
        df = df.copy()
        
        # Convert to lowercase and strip whitespace
        df[email_col] = df[email_col].str.lower().str.strip()
        
        # Email validation pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        # Create validation flag
        df['email_valid'] = df[email_col].apply(
            lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
        )
        
        # Extract domain for additional analysis
        df['email_domain'] = df[email_col].str.extract(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
        
        # Flag generic/personal emails
        generic_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        df['is_business_email'] = ~df['email_domain'].isin(generic_domains)
        
        valid_count = df['email_valid'].sum()
        logger.info(f"Valid emails: {valid_count}/{len(df)}")
        
        return df
    
    def clean_phone_numbers(self, df: pd.DataFrame, phone_col: str = 'phone') -> pd.DataFrame:
        """
        Clean and standardize phone numbers to E.164 format.
        
        Args:
            df: Input DataFrame
            phone_col: Name of phone column
            
        Returns:
            DataFrame with cleaned phone numbers
        """
        logger.info("Cleaning phone numbers...")
        
        df = df.copy()
        
        def standardize_phone(phone):
            """Standardize phone number format"""
            if pd.isna(phone):
                return None
            
            # Remove all non-numeric characters
            phone_clean = re.sub(r'\D', '', str(phone))
            
            # Handle different formats
            if len(phone_clean) == 10:  # US format without country code
                return f"+1{phone_clean}"
            elif len(phone_clean) == 11 and phone_clean.startswith('1'):
                return f"+{phone_clean}"
            elif len(phone_clean) >= 10:
                return f"+{phone_clean}"
            else:
                return None
        
        df[f'{phone_col}_cleaned'] = df[phone_col].apply(standardize_phone)
        df['phone_valid'] = df[f'{phone_col}_cleaned'].notna()
        
        valid_count = df['phone_valid'].sum()
        logger.info(f"Valid phones: {valid_count}/{len(df)}")
        
        return df
    
    def standardize_job_titles(self, df: pd.DataFrame, title_col: str = 'job_title') -> pd.DataFrame:
        """
        Standardize and categorize job titles into seniority levels.
        
        Args:
            df: Input DataFrame
            title_col: Name of job title column
            
        Returns:
            DataFrame with standardized job titles and seniority levels
        """
        logger.info("Standardizing job titles...")
        
        df = df.copy()
        
        # Clean job titles
        df[title_col] = df[title_col].str.strip().str.title()
        
        # Define seniority mapping
        def categorize_seniority(title):
            """Categorize job title into seniority level"""
            if pd.isna(title):
                return 'Unknown'
            
            title_lower = str(title).lower()
            
            # C-Level
            if any(x in title_lower for x in ['ceo', 'cto', 'cfo', 'coo', 'cmo', 'chief']):
                return 'C-Level'
            
            # VP/SVP Level
            elif any(x in title_lower for x in ['vp', 'vice president', 'svp']):
                return 'VP'
            
            # Director Level
            elif 'director' in title_lower:
                return 'Director'
            
            # Manager Level
            elif 'manager' in title_lower or 'head of' in title_lower:
                return 'Manager'
            
            # Senior Level
            elif any(x in title_lower for x in ['senior', 'sr.', 'lead']):
                return 'Senior'
            
            # Junior/Entry Level
            elif any(x in title_lower for x in ['junior', 'jr.', 'associate', 'coordinator']):
                return 'Junior'
            
            else:
                return 'Mid-Level'
        
        df['seniority_level'] = df[title_col].apply(categorize_seniority)
        
        # Categorize by department
        def categorize_department(title):
            """Categorize job title by department"""
            if pd.isna(title):
                return 'Unknown'
            
            title_lower = str(title).lower()
            
            if any(x in title_lower for x in ['sales', 'business development', 'account']):
                return 'Sales'
            elif any(x in title_lower for x in ['marketing', 'growth', 'brand']):
                return 'Marketing'
            elif any(x in title_lower for x in ['engineer', 'developer', 'technical', 'software']):
                return 'Engineering'
            elif any(x in title_lower for x in ['product', 'pm']):
                return 'Product'
            elif any(x in title_lower for x in ['hr', 'human resources', 'people']):
                return 'HR'
            elif any(x in title_lower for x in ['finance', 'accounting', 'financial']):
                return 'Finance'
            elif any(x in title_lower for x in ['operation', 'ops']):
                return 'Operations'
            else:
                return 'Other'
        
        df['department'] = df[title_col].apply(categorize_department)
        
        logger.info(f"Categorized {len(df)} job titles")
        
        return df
    
    def standardize_company_names(self, df: pd.DataFrame, company_col: str = 'company_name') -> pd.DataFrame:
        """
        Standardize company names by removing common suffixes and cleaning.
        
        Args:
            df: Input DataFrame
            company_col: Name of company column
            
        Returns:
            DataFrame with standardized company names
        """
        logger.info("Standardizing company names...")
        
        df = df.copy()
        
        def clean_company_name(name):
            """Clean and standardize company name"""
            if pd.isna(name):
                return None
            
            name = str(name).strip()
            
            # Remove common suffixes
            suffixes = [
                r'\s+Inc\.?$', r'\s+LLC\.?$', r'\s+Ltd\.?$', r'\s+Corp\.?$',
                r'\s+Corporation$', r'\s+Limited$', r'\s+Company$', r'\s+Co\.?$',
                r'\s+LLP$', r'\s+LP$', r'\s+PLC$', r'\s+Pvt\.?\s+Ltd\.?$'
            ]
            
            for suffix in suffixes:
                name = re.sub(suffix, '', name, flags=re.IGNORECASE)
            
            # Clean whitespace
            name = ' '.join(name.split())
            
            return name
        
        df[f'{company_col}_cleaned'] = df[company_col].apply(clean_company_name)
        
        logger.info(f"Standardized {len(df)} company names")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
        """
        Remove duplicate records based on key columns.
        
        Args:
            df: Input DataFrame
            key_columns: Columns to use for duplicate detection
            
        Returns:
            DataFrame with duplicates removed
        """
        logger.info(f"Removing duplicates based on: {key_columns}")
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=key_columns, keep='first')
        
        duplicates_removed = initial_count - len(df)
        self.stats['duplicate_records'] = duplicates_removed
        
        logger.info(f"Removed {duplicates_removed} duplicate records")
        
        return df
    
    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich data with additional computed fields.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with enriched data
        """
        logger.info("Enriching data...")
        
        df = df.copy()
        
        # Add data quality score
        df['data_quality_score'] = (
            df.get('email_valid', False).astype(int) * 30 +
            df.get('phone_valid', False).astype(int) * 25 +
            df.get('is_business_email', False).astype(int) * 20 +
            (~df['company_name_cleaned'].isna()).astype(int) * 15 +
            (~df['seniority_level'].isin(['Unknown'])).astype(int) * 10
        )
        
        # Add timestamp
        df['processed_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add data completeness percentage
        total_fields = len(df.columns)
        df['data_completeness'] = df.notna().sum(axis=1) / total_fields * 100
        
        logger.info("Data enrichment completed")
        
        return df
    
    def run_pipeline(self, input_file: str, output_file: str) -> pd.DataFrame:
        """
        Run the complete data pipeline.
        
        Args:
            input_file: Path to input data file
            output_file: Path to save cleaned data
            
        Returns:
            Cleaned and processed DataFrame
        """
        logger.info("Starting B2B Contact Data Pipeline...")
        
        # Load data
        df = self.load_data(input_file)
        
        # Clean and validate emails
        df = self.clean_email_addresses(df)
        
        # Clean and standardize phone numbers
        df = self.clean_phone_numbers(df)
        
        # Standardize job titles
        df = self.standardize_job_titles(df)
        
        # Standardize company names
        df = self.standardize_company_names(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df, key_columns=['email', 'company_name_cleaned'])
        
        # Enrich data
        df = self.enrich_data(df)
        
        # Update statistics
        self.stats['cleaned_records'] = len(df[df['data_quality_score'] >= 50])
        self.stats['invalid_records'] = len(df[df['data_quality_score'] < 50])
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        logger.info(f"Cleaned data saved to {output_file}")
        
        # Print statistics
        self.print_statistics()
        
        return df
    
    def print_statistics(self):
        """Print pipeline statistics"""
        print("\n" + "="*60)
        print("B2B CONTACT DATA PIPELINE - STATISTICS")
        print("="*60)
        print(f"Total Records Processed: {self.stats['total_records']}")
        print(f"High Quality Records: {self.stats['cleaned_records']}")
        print(f"Low Quality Records: {self.stats['invalid_records']}")
        print(f"Duplicates Removed: {self.stats['duplicate_records']}")
        print("="*60 + "\n")


def generate_sample_data(output_file: str = 'sample_contacts.csv', num_records: int = 1000):
    """
    Generate sample B2B contact data for testing the pipeline.
    
    Args:
        output_file: Path to save sample data
        num_records: Number of sample records to generate
    """
    np.random.seed(42)
    
    # Sample data
    first_names = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Emily', 'Robert', 'Lisa']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller']
    companies = ['TechCorp', 'DataSystems', 'CloudSolutions', 'InnovateLabs', 'DigitalVentures']
    domains = ['techcorp.com', 'datasystems.io', 'cloudsolutions.net', 'innovatelabs.com']
    job_titles = [
        'CEO', 'CTO', 'VP of Sales', 'Director of Marketing', 'Senior Engineer',
        'Sales Manager', 'Product Manager', 'Business Development Manager',
        'Head of Operations', 'Chief Financial Officer'
    ]
    
    data = []
    for i in range(num_records):
        first_name = np.random.choice(first_names)
        last_name = np.random.choice(last_names)
        company = np.random.choice(companies)
        domain = np.random.choice(domains)
        
        # Add some noise to data
        email = f"{first_name.lower()}.{last_name.lower()}@{domain}"
        if np.random.random() > 0.9:  # 10% invalid emails
            email = email.replace('@', '')
        
        phone = f"+1{np.random.randint(200, 999)}{np.random.randint(100, 999)}{np.random.randint(1000, 9999)}"
        if np.random.random() > 0.85:  # 15% invalid phones
            phone = phone[:5]
        
        record = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'phone': phone,
            'company_name': f"{company} Inc.",
            'job_title': np.random.choice(job_titles),
            'location': np.random.choice(['San Francisco', 'New York', 'Seattle', 'Boston'])
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info(f"Generated {num_records} sample records in {output_file}")


if __name__ == "__main__":
    # Generate sample data
    generate_sample_data('sample_contacts.csv', 1000)
    
    # Run pipeline
    pipeline = B2BContactDataPipeline()
    cleaned_df = pipeline.run_pipeline('sample_contacts.csv', 'cleaned_contacts.csv')
    
    # Display sample of cleaned data
    print("\nSample of Cleaned Data:")
    print(cleaned_df[['email', 'phone_cleaned', 'company_name_cleaned', 
                      'seniority_level', 'department', 'data_quality_score']].head(10))
