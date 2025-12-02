"""
Enhanced Feature Extraction for Phishing Detection
FIXED VERSION - Properly detects long numeric subdomains
"""

import re
import urllib.parse
from typing import Dict
import tldextract
import math
from collections import Counter


class CompleteFeatureExtractor:
    """Extract comprehensive features for accurate phishing detection."""
    
    def __init__(self):
        self.brand_keywords = [
            'paypal','ebay','amazon','microsoft','apple','google','facebook',
            'instagram','linkedin','netflix','adobe','dropbox','twitter',
            'coinbase','binance','blockchain','wallet','bank','wellsfargo',
            'chase','citibank','americanexpress','visa','mastercard','yahoo',
            'outlook','icloud','gmail','whatsapp','telegram'
        ]

        self.suspicious_keywords = [
            'verify','verification','confirm','update','secure','login',
            'signin','account','suspended','limited','expire','urgent',
            'immediate','action','required','alert','warning','locked',
            'unauthorized','unusual','activity','restore','reset','validate',
            'support','customer','service','help','billing'
        ]

        self.suspicious_tlds = {
            'tk','ml','ga','cf','gq','pw','cc','top','work','click','link',
            'download','zip','review','country','stream','gdn','mom','xin',
            'kim','men','loan','win','bid','racing','trade','xyz','club'
        }

        self.shorteners = {
            'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','short.link',
            'tiny.cc','is.gd','buff.ly','adf.ly','shorturl.at'
        }

    def _numeric_ratio(self, s: str) -> float:
        """Calculate ratio of numeric characters in string."""
        if not s:
            return 0.0
        return sum(c.isdigit() for c in s) / len(s)

    def _is_all_numeric(self, s: str) -> bool:
        """Check if string contains only digits."""
        return s.isdigit() if s else False

    def _entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        counter = Counter(text)
        length = len(text)
        return -sum((count/length) * math.log2(count/length)
                    for count in counter.values())

    def _consonant_ratio(self, text: str) -> float:
        """Calculate ratio of consonants to letters."""
        if not text:
            return 0.0
        letters = [c for c in text.lower() if c.isalpha()]
        if not letters:
            return 0.0
        vowels = "aeiou"
        consonants = [c for c in letters if c not in vowels]
        return len(consonants) / len(letters)

    def _is_ip(self, host: str) -> bool:
        """Check if hostname is an IP address."""
        import ipaddress
        try:
            ipaddress.ip_address(host)
            return True
        except:
            return False

    def extract_all_features(self, url: str) -> Dict[str, float]:
        """Extract all features from URL."""
        # Ensure URL has protocol
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        
        parsed = urllib.parse.urlparse(url)
        extracted = tldextract.extract(url)

        features = {}
        features.update(self._get_original_features(url))
        features.update(self._get_domain_features(parsed, extracted))
        features.update(self._get_brand_features(url.lower(), parsed, extracted))
        features.update(self._get_suspicious_patterns(url, parsed))
        features.update(self._get_entropy_features(url, parsed, extracted))
        features.update(self._get_security_features(parsed))
        
        return features

    def _get_original_features(self, url: str) -> Dict[str, float]:
        """Extract basic URL character features."""
        protocol_removed = re.sub(r'^https?://', '', url)
        n_redirection = protocol_removed.count('//')

        return {
            'url_length': float(len(url)),
            'n_dots': float(url.count('.')),
            'n_hypens': float(url.count('-')),
            'n_underline': float(url.count('_')),
            'n_slash': float(url.count('/')),
            'n_questionmark': float(url.count('?')),
            'n_equal': float(url.count('=')),
            'n_at': float(url.count('@')),
            'n_and': float(url.count('&')),
            'n_exclamation': float(url.count('!')),
            'n_space': float(url.count(' ')),
            'n_tilde': float(url.count('~')),
            'n_comma': float(url.count(',')),
            'n_plus': float(url.count('+')),
            'n_asterisk': float(url.count('*')),
            'n_hastag': float(url.count('#')),
            'n_dollar': float(url.count('$')),
            'n_percent': float(url.count('%')),
            'n_redirection': float(n_redirection)
        }

    def _get_domain_features(self, parsed, extracted) -> Dict[str, float]:
        """Extract domain and subdomain features - CRITICAL FOR PHISHING DETECTION."""
        domain = extracted.domain.lower()
        subdomain = extracted.subdomain.lower()
        tld = extracted.suffix.lower()
        hostname = parsed.netloc.lower().split(':')[0]

        is_ip = self._is_ip(hostname)
        tld_main = tld.split('.')[-1] if tld else ""
        is_suspicious_tld = tld_main in self.suspicious_tlds
        sub_parts = subdomain.split('.') if subdomain else []

        # Clean subdomain (remove dots for analysis)
        sub_raw = subdomain.replace('.', '')

        # 🔥 CRITICAL PHISHING INDICATORS - Fixed calculations
        sub_numeric_ratio = self._numeric_ratio(sub_raw)
        sub_is_numeric_only = float(self._is_all_numeric(sub_raw))
        sub_entropy = self._entropy(sub_raw)
        
        # IMPORTANT: These are the key features for detecting your phishing URLs
        # A subdomain that is >70% numeric AND longer than 10 chars is highly suspicious
        long_numeric_sub = float(sub_numeric_ratio > 0.7 and len(sub_raw) > 10)
        
        # Random-looking numeric subdomains (low entropy but high numeric content)
        random_numeric_sub = float(sub_entropy < 3.0 and sub_numeric_ratio > 0.8 and len(sub_raw) > 8)
        
        # SUPER SUSPICIOUS: Very long (>20 chars) numeric subdomains
        very_long_numeric_sub = float(len(sub_raw) > 20 and sub_numeric_ratio > 0.6)

        return {
            'domain_length': float(len(domain)),
            'subdomain_length': float(len(subdomain)),
            'subdomain_count': float(len(sub_parts)),
            'is_ip_address': float(is_ip),
            'suspicious_tld': float(is_suspicious_tld),
            'domain_has_digits': float(any(c.isdigit() for c in domain)),
            'tld_length': float(len(tld)),
            'hostname_length': float(len(hostname)),
            'has_port': float(':' in parsed.netloc),
            
            # 🚨 CRITICAL FEATURES FOR NUMERIC SUBDOMAIN DETECTION
            'subdomain_numeric_ratio': float(sub_numeric_ratio),
            'subdomain_is_numeric_only': float(sub_is_numeric_only),
            'subdomain_entropy': float(sub_entropy),
            'long_numeric_subdomain': float(long_numeric_sub),
            'random_numeric_subdomain': float(random_numeric_sub),
            'very_long_numeric_subdomain': float(very_long_numeric_sub),
            
            # Additional subdomain indicators
            'multiple_subdomains': float(len(sub_parts) > 1),
        }

    def _get_brand_features(self, url_lower, parsed, extracted):
        """Extract brand impersonation features."""
        domain = extracted.domain.lower()
        subdomain = extracted.subdomain.lower()
        path = parsed.path.lower()

        brand_in_sub = any(b in subdomain for b in self.brand_keywords)
        brand_in_dom = any(b in domain for b in self.brand_keywords)
        brand_in_path = any(b in path for b in self.brand_keywords)

        # Brand spoofing: brand in subdomain but NOT in main domain
        brand_spoof = brand_in_sub and not brand_in_dom
        
        # Count suspicious keywords
        susp_count = sum(1 for kw in self.suspicious_keywords if kw in url_lower)
        
        # Pure brand domain (legitimate pattern)
        is_pure_brand = brand_in_dom and not brand_in_sub

        return {
            'brand_in_subdomain': float(brand_in_sub),
            'brand_in_domain': float(brand_in_dom),
            'brand_in_path': float(brand_in_path),
            'brand_spoofing_pattern': float(brand_spoof),
            'suspicious_keyword_count': float(susp_count),
            'has_suspicious_keyword': float(susp_count > 0),
            'is_pure_brand_domain': float(is_pure_brand),
            'brand_and_suspicious': float(brand_spoof and susp_count > 0)
        }

    def _get_suspicious_patterns(self, url, parsed):
        """Extract suspicious URL patterns."""
        hostname = parsed.netloc.lower()
        path = parsed.path.lower()
        query = parsed.query.lower()

        path_depth = len([p for p in path.split('/') if p])
        is_shortener = any(s in hostname for s in self.shorteners)
        
        # Long subdomain
        subdomain_parts = hostname.split('.')
        first_part = subdomain_parts[0] if subdomain_parts else ""
        
        # Suspicious path keywords
        suspicious_path_keywords = ['login', 'signin', 'verify', 'account', 'update', 'secure']
        has_suspicious_path = any(k in path for k in suspicious_path_keywords)

        return {
            'is_url_shortener': float(is_shortener),
            'long_subdomain': float(len(first_part) > 20),
            'suspicious_path': float(has_suspicious_path),
            'has_php_extension': float('.php' in path),
            'path_depth': float(path_depth),
            'long_query_string': float(len(query) > 100),
            'has_multiple_dashes': float(hostname.count('-') > 3),
        }

    def _get_entropy_features(self, url, parsed, extracted):
        """Calculate entropy features for randomness detection."""
        domain = extracted.domain.lower()
        hostname = parsed.netloc.lower()
        subdomain = extracted.subdomain.lower()

        url_entropy = self._entropy(url)
        domain_entropy = self._entropy(domain)
        hostname_entropy = self._entropy(hostname)
        subdomain_entropy = self._entropy(subdomain) if subdomain else 0.0
        
        # High entropy indicates random/suspicious patterns
        high_entropy_host = float(hostname_entropy > 4.2)

        return {
            'url_entropy': url_entropy,
            'domain_entropy': domain_entropy,
            'hostname_entropy': hostname_entropy,
            'high_entropy_hostname': high_entropy_host,
            'domain_consonant_ratio': self._consonant_ratio(domain),
        }

    def _get_security_features(self, parsed):
        """Extract security-related features."""
        return {
            'is_https': float(parsed.scheme == 'https'),
            'is_http': float(parsed.scheme == 'http'),
            'has_query_params': float(bool(parsed.query)),
            'no_protocol': 0.0  # We always add protocol, so this is always 0
        }

    def get_feature_names(self):
        """Get list of all feature names in correct order."""
        return list(self.extract_all_features("http://example.com").keys())


def extract_enhanced_features(url: str) -> Dict[str, float]:
    """Convenience function to extract features."""
    return CompleteFeatureExtractor().extract_all_features(url)


# Test the extractor with your phishing URLs
if __name__ == "__main__":
    extractor = CompleteFeatureExtractor()
    
    test_urls = [
        "http://00000000883838383992929292222.ratingandreviews.in",
        "http://00000000000000000000000000000000000000000.xyz",
        "https://www.google.com",
        "http://paypal-verify.suspicious.com"
    ]
    
    print("=" * 80)
    print("TESTING FEATURE EXTRACTION")
    print("=" * 80)
    
    for url in test_urls:
        print(f"\n🔍 URL: {url}")
        features = extractor.extract_all_features(url)
        
        # Print key phishing indicators
        print(f"   Subdomain length: {features['subdomain_length']:.0f}")
        print(f"   Numeric ratio: {features['subdomain_numeric_ratio']:.2f}")
        print(f"   Is all numeric: {features['subdomain_is_numeric_only']:.0f}")
        print(f"   Long numeric subdomain: {features['long_numeric_subdomain']:.0f}")
        print(f"   Very long numeric subdomain: {features['very_long_numeric_subdomain']:.0f}")
        print(f"   Suspicious TLD: {features['suspicious_tld']:.0f}")
        print(f"   HTTPS: {features['is_https']:.0f}")
        
        # Verdict based on features
        if features['very_long_numeric_subdomain'] == 1.0:
            print("   ⚠️ VERDICT: HIGHLY SUSPICIOUS - Very long numeric subdomain")
        elif features['long_numeric_subdomain'] == 1.0:
            print("   ⚠️ VERDICT: SUSPICIOUS - Long numeric subdomain")
        else:
            print("   ✓ VERDICT: No obvious numeric subdomain issues")