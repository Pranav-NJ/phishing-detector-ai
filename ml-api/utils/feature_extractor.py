"""
URL Feature Extraction for Phishing Detection

This module extracts various features from URLs that are commonly used
to identify phishing websites. Each feature is carefully chosen based on
research showing its effectiveness in distinguishing legitimate from malicious URLs.
"""

import re
import urllib.parse
import tldextract
from typing import Dict, List
import ipaddress


class URLFeatureExtractor:
    """Extract features from URLs for phishing detection."""
    
    def __init__(self):
        # Common phishing keywords that often appear in malicious URLs
        self.phishing_keywords = [
            'chatgpt', 'paypal', 'ebay', 'amazon', 'microsoft', 'apple', 'google', 'facebook',
            'twitter', 'instagram', 'linkedin', 'netflix', 'adobe', 'dropbox',
            'login', 'signin', 'account', 'verify', 'secure', 'update', 'confirm',
            'suspended', 'limited', 'expired', 'urgent', 'immediate', 'action',
            'click', 'here', 'now', 'today', 'free', 'winner', 'congratulations'
        ]
        
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = [
            'tk', 'ml', 'ga', 'cf', 'click', 'link', 'download', 'zip', 'review'
        ]
    
    def extract_all_features(self, url: str) -> Dict[str, float]:
        """
        Extract all features from a URL.
        
        Args:
            url (str): The URL to analyze
            
        Returns:
            Dict[str, float]: Dictionary of feature names and values
        """
        try:
            parsed = urllib.parse.urlparse(url)
            extracted = tldextract.extract(url)
            
            features = {}
            
            # Basic URL structure features
            features.update(self._get_length_features(url, parsed))
            features.update(self._get_character_features(url, parsed))
            features.update(self._get_domain_features(parsed, extracted))
            features.update(self._get_path_features(parsed))
            features.update(self._get_security_features(parsed))
            features.update(self._get_suspicious_features(url, parsed))
            features.update(self._get_phishing_keyword_features(url.lower()))
            
            return features
            
        except Exception as e:
            # Return default features if URL parsing fails
            return self._get_default_features()
    
    def _get_length_features(self, url: str, parsed) -> Dict[str, float]:
        """Extract length-based features from URL."""
        return {
            'url_length': len(url),
            'hostname_length': len(parsed.netloc),
            'path_length': len(parsed.path),
            'query_length': len(parsed.query),
            'fragment_length': len(parsed.fragment)
        }
    
    def _get_character_features(self, url: str, parsed) -> Dict[str, float]:
        """
        Extract character-based features that indicate suspicious URLs.
        
        Phishing URLs often contain unusual character patterns:
        - Multiple dots (to mimic legitimate subdomains)
        - Dashes (to create confusing domain names)
        - At symbols (used in phishing techniques)
        - Digits in domain (legitimate sites rarely have numbers)
        """
        hostname = parsed.netloc.lower()
        
        return {
            'dots_count': url.count('.'),
            'dashes_count': url.count('-'),
            'underscores_count': url.count('_'),
            'slashes_count': url.count('/'),
            'question_marks_count': url.count('?'),
            'equals_count': url.count('='),
            'at_symbol_count': url.count('@'),
            'ampersand_count': url.count('&'),
            'digits_count': sum(c.isdigit() for c in url),
            'digits_in_hostname': sum(c.isdigit() for c in hostname),
            'special_chars_count': len(re.findall(r'[^a-zA-Z0-9./\-_?=&]', url))
        }
    
    def _get_domain_features(self, parsed, extracted) -> Dict[str, float]:
        """
        Extract domain-related features.
        
        Domain features help identify:
        - IP addresses instead of domain names (suspicious)
        - Suspicious TLDs commonly used in phishing
        - Subdomain patterns used to deceive users
        """
        hostname = parsed.netloc.lower()
        domain = extracted.domain.lower()
        subdomain = extracted.subdomain.lower()
        tld = extracted.suffix.lower()
        
        # Check if hostname is an IP address
        is_ip = self._is_ip_address(hostname.split(':')[0])  # Remove port if present
        
        return {
            'is_ip_address': float(is_ip),
            'subdomain_count': len(subdomain.split('.')) if subdomain else 0,
            'domain_length': len(domain),
            'subdomain_length': len(subdomain),
            'tld_length': len(tld),
            'suspicious_tld': float(tld in self.suspicious_tlds),
            'has_port': float(':' in parsed.netloc),
            'www_prefix': float(subdomain.startswith('www'))
        }
    
    def _get_path_features(self, parsed) -> Dict[str, float]:
        """
        Extract path-related features.
        
        Path analysis helps identify:
        - Deep directory structures (often used to hide malicious content)
        - Suspicious file extensions
        - URL redirection patterns
        """
        path = parsed.path.lower()
        
        return {
            'path_segments_count': len([seg for seg in path.split('/') if seg]),
            'has_extension': float('.' in path.split('/')[-1] if path else False),
            'path_contains_exe': float('.exe' in path),
            'path_contains_zip': float('.zip' in path),
            'path_contains_admin': float('admin' in path),
            'path_contains_login': float('login' in path or 'signin' in path),
            'double_slash_in_path': float('//' in path),
        }
    
    def _get_security_features(self, parsed) -> Dict[str, float]:
        """
        Extract security-related features.
        
        Security features include:
        - HTTPS usage (legitimate sites increasingly use HTTPS)
        - Port usage (non-standard ports can be suspicious)
        """
        return {
            'is_https': float(parsed.scheme == 'https'),
            'is_http': float(parsed.scheme == 'http'),
            'has_query': float(bool(parsed.query)),
            'has_fragment': float(bool(parsed.fragment))
        }
    
    def _get_suspicious_features(self, url: str, parsed) -> Dict[str, float]:
        """
        Extract features that directly indicate suspicious patterns.
        
        These patterns are commonly found in phishing URLs:
        - URL shorteners (used to hide real destination)
        - Suspicious character combinations
        - Deceptive formatting
        """
        url_lower = url.lower()
        hostname = parsed.netloc.lower()
        
        # Common URL shorteners
        shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'short.link']
        
        return {
            'url_shortener': float(any(short in hostname for short in shorteners)),
            'contains_ip_pattern': float(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
            'multiple_subdomains': float(hostname.count('.') > 2),
            'suspicious_words': float(any(word in url_lower for word in 
                                        ['click', 'here', 'now', 'urgent', 'verify', 'suspended'])),
            'url_encoding': float('%' in url),
            'long_subdomain': float(len(parsed.netloc.split('.')[0]) > 20 if '.' in parsed.netloc else False)
        }
    
    def _get_phishing_keyword_features(self, url_lower: str) -> Dict[str, float]:
        """
        Count occurrences of common phishing keywords.
        
        Phishing sites often impersonate legitimate services by including
        brand names and action words in their URLs.
        """
        features = {}
        
        # Count brand names (companies often targeted by phishing)
        brand_keywords = ['paypal', 'ebay', 'amazon', 'microsoft', 'apple', 'google', 'facebook']
        features['brand_keywords_count'] = sum(url_lower.count(keyword) for keyword in brand_keywords)
        
        # Count action words (used to create urgency)
        action_keywords = ['login', 'signin', 'verify', 'update', 'confirm', 'secure']
        features['action_keywords_count'] = sum(url_lower.count(keyword) for keyword in action_keywords)
        
        # Count urgency words
        urgency_keywords = ['urgent', 'immediate', 'suspended', 'expired', 'limited']
        features['urgency_keywords_count'] = sum(url_lower.count(keyword) for keyword in urgency_keywords)
        
        return features
    
    def _is_ip_address(self, hostname: str) -> bool:
        """Check if hostname is an IP address."""
        try:
            ipaddress.ip_address(hostname)
            return True
        except ValueError:
            return False
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when URL parsing fails."""
        return {
            'url_length': 0,
            'hostname_length': 0,
            'path_length': 0,
            'query_length': 0,
            'fragment_length': 0,
            'dots_count': 0,
            'dashes_count': 0,
            'underscores_count': 0,
            'slashes_count': 0,
            'question_marks_count': 0,
            'equals_count': 0,
            'at_symbol_count': 0,
            'ampersand_count': 0,
            'digits_count': 0,
            'digits_in_hostname': 0,
            'special_chars_count': 0,
            'is_ip_address': 0,
            'subdomain_count': 0,
            'domain_length': 0,
            'subdomain_length': 0,
            'tld_length': 0,
            'suspicious_tld': 0,
            'has_port': 0,
            'www_prefix': 0,
            'path_segments_count': 0,
            'has_extension': 0,
            'path_contains_exe': 0,
            'path_contains_zip': 0,
            'path_contains_admin': 0,
            'path_contains_login': 0,
            'double_slash_in_path': 0,
            'is_https': 0,
            'is_http': 0,
            'has_query': 0,
            'has_fragment': 0,
            'url_shortener': 0,
            'contains_ip_pattern': 0,
            'multiple_subdomains': 0,
            'suspicious_words': 0,
            'url_encoding': 0,
            'long_subdomain': 0,
            'brand_keywords_count': 0,
            'action_keywords_count': 0,
            'urgency_keywords_count': 0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return list(self._get_default_features().keys())


def extract_features_from_url(url: str) -> Dict[str, float]:
    """
    Convenience function to extract features from a single URL.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        Dict[str, float]: Dictionary of features
    """
    extractor = URLFeatureExtractor()
    return extractor.extract_all_features(url)