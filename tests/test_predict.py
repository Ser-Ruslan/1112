"""
Unit tests for regression model prediction
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model import RegressionModel


class TestRegressionModel(unittest.TestCase):
    """Test cases for regression model"""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests"""
        try:
            cls.model = RegressionModel()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found: {e}")
    
    def test_model_loaded_successfully(self):
        """Test that model loads without errors"""
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.scaler)
        self.assertIsNotNone(self.model.config)
    
    def test_model_has_correct_config(self):
        """Test that model config has required fields"""
        required_fields = ['model_type', 'target_variable', 'intercept', 
                          'coefficients', 'test_r2']
        
        for field in required_fields:
            self.assertIn(field, self.model.config)
    
    def test_prediction_output_format(self):
        """Test that prediction returns expected format"""
        result = self.model.predict(
            age=55,
            trestbps=120,
            chol=200,
            oldpeak=1.5,
            sex='male',
            cp='asympt',
            fbs='false',
            restecg='norm',
            exang='false',
            slope='flat',
            thal='norm'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('features_dict', result)
        self.assertIsInstance(result['prediction'], float)
    
    def test_prediction_in_reasonable_range(self):
        """Test that predictions are in reasonable range for heart rate"""
        result = self.model.predict(
            age=55,
            trestbps=120,
            chol=200,
            oldpeak=1.5,
            sex='male',
            cp='asympt',
            fbs='false',
            restecg='norm',
            exang='false',
            slope='flat',
            thal='norm'
        )
        
        prediction = result['prediction']
        # Heart rate should be between 60 and 220 bpm
        self.assertGreater(prediction, 60)
        self.assertLess(prediction, 220)
    
    def test_prediction_varies_with_age(self):
        """Test that predictions change with different ages"""
        result_young = self.model.predict(
            age=30,
            trestbps=120,
            chol=200,
            oldpeak=1.5,
            sex='male',
            cp='asympt',
            fbs='false',
            restecg='norm',
            exang='false',
            slope='flat',
            thal='norm'
        )
        
        result_old = self.model.predict(
            age=70,
            trestbps=120,
            chol=200,
            oldpeak=1.5,
            sex='male',
            cp='asympt',
            fbs='false',
            restecg='norm',
            exang='false',
            slope='flat',
            thal='norm'
        )
        
        # Predictions should differ
        self.assertNotEqual(result_young['prediction'], result_old['prediction'])
        
        # Older age should generally lead to lower max heart rate
        self.assertLess(result_old['prediction'], result_young['prediction'])
    
    def test_get_model_info(self):
        """Test get_model_info method"""
        info = self.model.get_model_info()
        
        self.assertIn('model_type', info)
        self.assertIn('target_variable', info)
        self.assertIn('intercept', info)
        self.assertIn('coefficients', info)
        self.assertEqual(info['model_type'], 'LinearRegression')
        self.assertEqual(info['target_variable'], 'thalach')
    
    def test_get_r2_score(self):
        """Test get_r2_score method"""
        r2 = self.model.get_r2_score()
        
        self.assertIsInstance(r2, float)
        self.assertGreaterEqual(r2, 0)
        self.assertLessEqual(r2, 1)
    
    def test_get_equation_latex(self):
        """Test get_equation_latex method"""
        equation = self.model.get_equation_latex()
        
        self.assertIsInstance(equation, str)
        self.assertIn('$', equation)
        self.assertIn('y =', equation)
    
    def test_prepare_features(self):
        """Test _prepare_features method"""
        features = self.model._prepare_features(
            age=55,
            trestbps=120,
            chol=200,
            oldpeak=1.5,
            sex='male',
            cp='asympt',
            fbs='false',
            restecg='norm',
            exang='false',
            slope='flat',
            thal='norm'
        )
        
        self.assertEqual(len(features), len(self.model.feature_columns))
        self.assertIsInstance(features, list)
        # Check that all features are numeric
        for feat in features:
            self.assertIsInstance(feat, (int, float))
    
    def test_reproducibility(self):
        """Test that same input produces same output"""
        result1 = self.model.predict(
            age=55,
            trestbps=120,
            chol=200,
            oldpeak=1.5,
            sex='male',
            cp='asympt',
            fbs='false',
            restecg='norm',
            exang='false',
            slope='flat',
            thal='norm'
        )
        
        result2 = self.model.predict(
            age=55,
            trestbps=120,
            chol=200,
            oldpeak=1.5,
            sex='male',
            cp='asympt',
            fbs='false',
            restecg='norm',
            exang='false',
            slope='flat',
            thal='norm'
        )
        
        self.assertEqual(result1['prediction'], result2['prediction'])
    
    def test_categorical_encoding(self):
        """Test that different categorical values are encoded differently"""
        result_male = self.model.predict(
            age=55, trestbps=120, chol=200, oldpeak=1.5,
            sex='male', cp='asympt', fbs='false', restecg='norm',
            exang='false', slope='flat', thal='norm'
        )
        
        result_female = self.model.predict(
            age=55, trestbps=120, chol=200, oldpeak=1.5,
            sex='female', cp='asympt', fbs='false', restecg='norm',
            exang='false', slope='flat', thal='norm'
        )
        
        # Different sex should produce different predictions
        self.assertNotEqual(result_male['prediction'], result_female['prediction'])


class TestModelConsistency(unittest.TestCase):
    """Test model consistency and integrity"""
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests"""
        cls.model = RegressionModel()
    
    def test_feature_columns_match_config(self):
        """Test that feature columns match config"""
        self.assertEqual(
            self.model.feature_columns,
            self.model.config['feature_columns']
        )
    
    def test_numeric_features_in_features(self):
        """Test that numeric features are in feature columns"""
        for feat in self.model.numeric_features:
            self.assertIn(feat, self.model.feature_columns)
    
    def test_categorical_features_encoded(self):
        """Test that categorical features are properly encoded"""
        for cat_feat in self.model.categorical_features:
            encoded_feat = f'{cat_feat}_encoded'
            self.assertIn(encoded_feat, self.model.feature_columns)
    
    def test_coefficients_match_features(self):
        """Test that number of coefficients matches features"""
        self.assertEqual(
            len(self.model.config['coefficients']),
            len(self.model.feature_columns)
        )


if __name__ == '__main__':
    unittest.main()
