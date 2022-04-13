from unittest import TestCase
import unittest
import app

class MyTest(TestCase):
    def test_img_path_list(self):
        a = app.img_path_list([[0, 1, 2, 3, 4, 5]], 'bag')
        ans_a = ['train_image_data/bag/10.png',
                 'train_image_data/bag/11.png',
                 'train_image_data/bag/12.png',
                 'train_image_data/bag/13.png',
                 'train_image_data/bag/14.png']
        self.assertEqual(a, ans_a)

if __name__ == '__main__':
    unittest.main()