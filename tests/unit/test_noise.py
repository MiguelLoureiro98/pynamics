import unittest
from pynamics._noise._noise_generators import _white_noise

"""
This file contains unit tests targetting the white noise generating functions.
"""

class TestNoise(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        
        """
        Set up the necessary variables.
        """

        cls.length = 100000;
        
        cls.sample1 = _white_noise(1, cls.length, 1, 0);
        cls.sample2 = _white_noise(5, cls.length, 1, 0);
        cls.sample3 = _white_noise(1, cls.length, 100, 0);

    @classmethod
    def tearDownClass(cls) -> None:
        
        """
        Delete the variables created by the setUpClass method.
        """
        
        del cls.length;
        del cls.sample1;
        del cls.sample2;
        del cls.sample3;
    
        print("Sucessfully deleted all class attributes created by setUpClass.");

    def test_shape(self) -> None:
        """
        Verify that the shape of the noise arrays is as expected.
        """

        self.assertEqual(self.sample1.shape[1], self.length);
        self.assertEqual(self.sample2.shape[1], self.length);
        self.assertEqual(self.sample3.shape[1], self.length);
    
        self.assertEqual(self.sample1.shape[0], 1);
        self.assertEqual(self.sample2.shape[0], 5);
        self.assertEqual(self.sample3.shape[0], 1);

    def test_mean(self) -> None:
        """
        Test whether the mean of the noise arrays is close enough to zero.
        """

        self.assertAlmostEqual(self.sample1.mean(), 0.0, 2);
        self.assertAlmostEqual(self.sample2.mean(axis=1)[0], 0.0, 2);
        self.assertAlmostEqual(self.sample2.mean(axis=1)[1], 0.0, 2);
        self.assertAlmostEqual(self.sample2.mean(axis=1)[2], 0.0, 2);
        self.assertAlmostEqual(self.sample2.mean(axis=1)[3], 0.0, 2);
        self.assertAlmostEqual(self.sample2.mean(axis=1)[4], 0.0, 2);
        self.assertAlmostEqual(self.sample3.mean(), 0.0, 1);

    def test_variance(self) -> None:
        """
        Test whether the variance of the noise arrays are close enough to the one specified by the user.
        """

        self.assertAlmostEqual(self.sample1.var(), 1.0, 3);
        self.assertAlmostEqual(self.sample2.var(axis=1)[0], 1.0, 3);
        self.assertAlmostEqual(self.sample2.var(axis=1)[1], 1.0, 3);
        self.assertAlmostEqual(self.sample2.var(axis=1)[2], 1.0, 3);
        self.assertAlmostEqual(self.sample2.var(axis=1)[3], 1.0, 3);
        self.assertAlmostEqual(self.sample2.var(axis=1)[4], 1.0, 3);
        self.assertAlmostEqual(self.sample3.var(), 100.0, 1);

if __name__ == "__main__":

    unittest.main();