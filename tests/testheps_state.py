import unittest

from heps_state import *

class TestState(unittest.TestCase):
    def test_ideal_state(self):
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
        
        result = define_state(settings['filter_range'][0],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
        test_state = -np.round(result.full(), decimals=7)
        ideal_state = np.zeros((16,1), dtype=np.complex)
        ideal_state[3] = 0.5
        ideal_state[6] = 0.5
        ideal_state[9] = 0.5
        ideal_state[12] = 0.5
        self.assertEqual(test_state.tolist(), ideal_state.tolist())

    def test_nonideal_state(self):
        settings = {
            'lambda_deg':   1556e-9,
            'l1':           1.000,
            'l1_p':         1.020,
            'l2':           1.000,
            'l2_p':         1.040,                                  
            'a1':           95.0 * np.pi/180.0,                      
            'a2':           70.0 * np.pi/180.0,
            'p':            np.sqrt(0.4),                            
            'filter_range': np.linspace(0.5,1.0,1)*1e12      
        }
        
        result = define_state(settings['filter_range'][0],settings['a1'],settings['a2'],settings['l1'],settings['l1_p'],settings['l2'],settings['l2_p'])
        test_state = -np.round(result.full(), decimals=7)
        ideal_state = np.zeros((16,1), dtype=np.complex)
        ideal_state[3] = (0.4959639-0.0368831j)
        ideal_state[6] = (0.4959639+0.0368831j)
        ideal_state[9] = (0.071835+0.4565073j)
        ideal_state[12] = (0.1097079+0.4489135j)
        self.assertEqual(test_state.tolist(), ideal_state.tolist())


if __name__ == '__main__':
    unittest.main()