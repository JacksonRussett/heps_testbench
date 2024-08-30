import unittest

from heps_static_sim import *

class TestMismatch(unittest.TestCase):
    def test_nolength_mismatch_inft(self):
        settings = {
            'lambda_deg':   1556e-9,
            'l1':           1.000,
            'l1_p':         1.000,
            'l2':           1.000,
            'l2_p':         1.000,                                  
            'a1':           90.0 * np.pi/180.0,                      
            'a2':           90.0 * np.pi/180.0,
            'p':            np.sqrt(0.5),                            
            'filter_range': np.linspace(0.0,1.0,1)*1e12      
        }
        
        result = conc_length_mismatch(settings, 0.0, 0.0)
        self.assertEqual(np.round(result, decimals=7), 1.0)

    def test_length_mismatch_inft(self):
        settings = {
            'lambda_deg':   1556e-9,
            'l1':           1.000,
            'l1_p':         1.000,
            'l2':           1.000,
            'l2_p':         1.000,                                  
            'a1':           90.0 * np.pi/180.0,                      
            'a2':           90.0 * np.pi/180.0,
            'p':            np.sqrt(0.5),                            
            'filter_range': np.linspace(0.5,1.0,1)*1e12      
        }
        
        result = conc_length_mismatch(settings, 0.1, 0.1)
        self.assertEqual(np.round(result, decimals=7), 0.6856621)

    def test_noangle_mismatch_inft(self):
        settings = {
            'lambda_deg':   1556e-9,
            'l1':           1.000,
            'l1_p':         1.000,
            'l2':           1.000,
            'l2_p':         1.000,                                  
            'a1':           90.0 * np.pi/180.0,                      
            'a2':           90.0 * np.pi/180.0,
            'p':            np.sqrt(0.5),                            
            'filter_range': np.linspace(0.0,1.0,1)*1e12      
        }
        
        result = conc_angle_mismatch(settings, 0.0, 0.0)
        self.assertEqual(np.round(result, decimals=7), 1.0)

    def test_angle_mismatch_inft(self):
        settings = {
            'lambda_deg':   1556e-9,
            'l1':           1.000,
            'l1_p':         1.000,
            'l2':           1.000,
            'l2_p':         1.000,                                  
            'a1':           90.0 * np.pi/180.0,                      
            'a2':           90.0 * np.pi/180.0,
            'p':            np.sqrt(0.5),                            
            'filter_range': np.linspace(0.0,1.0,1)*1e12      
        }
        
        result = conc_angle_mismatch(settings, 20.0, 20.0)
        self.assertEqual(np.round(result, decimals=7), 0.5868241)

class TestSimulation(unittest.TestCase):
    def test_single_setting_example(self):
        
        (conc_inft, conc_fint) = single_setting_example(N=20)
        self.assertEqual(np.round(conc_inft, decimals=7), 1.0000000)
        self.assertEqual(np.round(conc_fint, decimals=7), 0.9781535)

if __name__ == '__main__':
    unittest.main()