'''
Test suite containing a dummy test to make pytest pass when no tests are provided.
'''


def test_dummy():
    '''
    A dummy test to ensure pytest does not fail in a CI/CD pipeline due to no tests
    being ran.
    '''
    assert True
