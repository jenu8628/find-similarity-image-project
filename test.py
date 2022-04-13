from unittest import TestCase
import PredictClothesCategory
import unittest

class MyTest(TestCase):
    pre = PredictClothesCategory.Preprocessor()

    def test_generate_filepath(self):
        a = self.pre.generate_filepath('train', 'bag', 'bag_1')
        ans_a = './flask/train_image_data/bag/bag_1.png'
        self.assertEqual(a, ans_a)


    def test_request_image_file(self):
        a = self.pre.request_image_file('http://img1a.coupangcdn.com/image/vendor_inventory/48bc/e366f27008d9437c9869bd992449a64b296e1eae87d3d701b05b2a50b3a8.jpeg')
        self.assertIsNotNone(a)

if __name__ == "__main__":
    unittest.main()