# def test_erf_basic_sanity():
#     """
#     Taken from literature directory 'error_functions.pdf'
#     which is from U. Waterloo, Canada.
#     """
#     assert_equal(erf(0.0), 0.0000000000)
#     assert_equal(erf(0.5), 0.5204998778)
#     assert_equal(erf(1.0), 0.8427007929)
#     assert_equal(erf(1.5), 0.9661051465)
#     assert_equal(erf(2.0), 0.9953222650)
#     assert_equal(erf(2.5), 0.9995930480)
#     assert_equal(erf(3.0), 0.9999779095)
#     assert_equal(erf(3.5), 0.9999992569)
#     assert_equal(erf(4.0), 0.9999999846)
#     assert_equal(erf(4.5), 0.9999999998)


if __name__ == '__main__':
    import nose
    nose.runmodule()