import pytest
import striker
import torch


def test_from_file_basic(tmp_path):
    """ Test that we can load parameters from arbitrary files """
    with open(tmp_path / 'param.py', 'w') as f:
        f.writelines([
            'import striker\n',
            'params = striker.Parameters(a=1)\n',
        ])

    p = striker.Parameters.from_file(tmp_path / 'param.py')
    assert p.a == 1


def test_from_file_func(tmp_path):
    """ Test that we can load functions with arguments as parameters from arbitrary files """
    with open(tmp_path / 'param.py', 'w') as f:
        f.writelines([
            'import striker\n',
            'def params(a):\n',
            '\treturn striker.Parameters(a=a)\n',
        ])

    p = striker.Parameters.from_file(tmp_path / 'param.py', a=123)
    assert p.a == 123


def test_save_load(tmp_path):
    """ Test Saving and Loading """
    def get_param():
        param = striker.Parameters(a=1, _b=2)
        param.c = torch.nn.Conv2d(3, 32, 3, 1, 1)
        param._d = torch.nn.Conv2d(3, 32, 3, 1, 1)
        return param

    p1 = get_param()
    p1.save(tmp_path / 'param.state.pt')

    p2 = striker.Parameters(c=torch.nn.Conv2d(3, 32, 3, 1, 1))
    p2.load(tmp_path / 'param.state.pt')

    assert p2.a == p1.a
    assert (p2.c.weight == p1.c.weight).all()
    assert (p2.c.bias == p1.c.bias).all()

    assert 'b' in p1
    assert 'b' not in p2
    assert 'd' in p1
    assert 'd' not in p2


def test_reset():
    """ Test the reset method """
    p = striker.Parameters()
    for key, value in striker.Parameters._Parameters__automatic.items():
        assert p.get(key) == value
        setattr(p, key, None)
        assert p.get(key) != value

    p.reset()
    for key, value in striker.Parameters._Parameters__automatic.items():
        assert p.get(key) == value


def test_get():
    """ Test the get method """
    p = striker.Parameters(a={'value': [0, 1, 2]})
    assert p.get('a.value.1') == 1
    assert p.get('b') is None
    assert p.get('a.none', 123) == 123
    assert p.get('a.value.50') is None


def test_add():
    """ Test that adding Parameter objects together works """
    p1 = striker.Parameters(a=1)
    p1.batch = 10
    p2 = striker.Parameters(a=2, b=1)

    p3 = p1 + p2
    assert p3.batch == 10
    assert p3.a == 1
    assert p3.b == 1

    p4 = p2 + p1
    assert p4.batch == 0
    assert p4.a == 2
    assert p4.b == 1

    p1 += p2
    assert p1.batch == 10
    assert p1.a == 1
    assert p1.b == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Need CUDA to cast to another device')
def test_to():
    """ Test that the to method works correctly """
    p = striker.Parameters()
    p.a = torch.nn.Conv2d(3, 32, 3, 1, 1)
    assert p.a.weight.device.type == 'cpu'

    p.to('cuda')
    assert p.a.weight.device.type == 'cuda'
