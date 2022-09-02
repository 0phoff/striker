import striker
striker.core.HookParent.__hook_check__ = 'raise'


def test_api():
    """ Test general hook usage API """
    class Parent(striker.core.HookParent, hook_types={'a', 'b', 'c'}):
        def __init__(self):
            self.hooks.check()

            self.value_a = 0
            self.value_b = 0
            self.value_c = 0

        @striker.hooks.a
        def test_hook_type(self):
            self.value_a += 1

        @striker.hooks.b[5:50:10]
        def test_hook_index(self):
            self.value_b += 1

        @striker.hooks.c
        def test_args(self, a, *, b=None):
            self.value_c += a + b

        def run(self):
            self.hooks.run(type='a')
            assert self.value_a == 1
            assert self.value_b == 0
            assert self.value_c == 0

            self.hooks.run(type='b', index=0)
            assert self.value_b == 0
            self.hooks.run(type='b', index=5)
            assert self.value_b == 1
            self.hooks.run(type='b', index=22)
            assert self.value_b == 1
            self.hooks.run(type='b', index=25)
            assert self.value_b == 2
            self.hooks.run(type='b', index=55)
            assert self.value_a == 1
            assert self.value_b == 2
            assert self.value_c == 0

            self.hooks.run(type='c', args=[111], kwargs={'b': 222})
            assert self.value_a == 1
            assert self.value_b == 2
            assert self.value_c == 333

    p = Parent()
    p.run()


def test_runtime():
    """ Test that hooks defined at runtime work and are correctly bound """
    class Parent(striker.core.HookParent, hook_types={'a', 'b'}):
        def __init__(self):
            self.hooks.check()

            self.value_a = 0
            self.value_b = 0
            self.hooks.a(self.test_hook_1)
            self.hooks.b[::5](self.test_hook_2)

        def test_hook_1(self):
            self.value_a += 1

        def test_hook_2(self, extra):
            self.value_b += extra

        def run(self):
            self.hooks.run(type='a')
            assert self.value_a == 1
            assert self.value_b == 0

            self.hooks.run(type='b', index=3, args=[1])
            assert self.value_a == 1
            assert self.value_b == 0

            self.hooks.run(type='b', index=5, args=[2])
            assert self.value_a == 1
            assert self.value_b == 2

            self.hooks.run(type='b', index=10, args=[3])
            assert self.value_a == 1
            assert self.value_b == 5

            self.hooks.run(type='b', index=13, args=[4])
            assert self.value_a == 1
            assert self.value_b == 5

    p = Parent()
    p.run()


def test_disable():
    """ Test that disabling a hook works correctly """
    class Parent(striker.core.HookParent, hook_types={'a'}):
        def __init__(self):
            self.hooks.check()
            self.value = 0

        @striker.hooks.a
        def hook_a(self):
            self.value += 1

        def run(self):
            self.hooks.run(type='a')
            assert self.value == 1

            self.hook_a.enabled = False
            print(self.hook_a.enabled, self.hook_a, self)
            self.hooks.run(type='a')
            assert self.value == 1

            self.hook_a.enabled = True
            self.hooks.run(type='a')
            assert self.value == 2

    p = Parent()
    p.run()


def test_multiple_parents():
    """ Test that multiple parent instances work correctly """
    class Parent(striker.core.HookParent, hook_types={'a'}):
        def __init__(self):
            self.hooks.check()
            self.value = 0

        @striker.hooks.a
        def hook(self):
            self.value += 1

        def run(self):
            self.hooks.run(type='a')
            assert self.value == 1

            self.hooks.run(type='a')
            assert self.value == 2

    p1 = Parent()
    p2 = Parent()
    p1.run()
    p2.run()

    p3 = Parent()
    p3.run()


def test_parent_inheritance():
    """ Test that inheritance works for parent classes """
    class Parent(striker.core.HookParent, hook_types={'a'}):
        def __init__(self):
            self.hooks.check()
            self.value_parent = 0

        @striker.hooks.a
        def hook_parent(self):
            self.value_parent += 1

        def run(self):
            self.hooks.run(type='a')
            self.hooks.run(type='a')

    class Child(Parent, hook_types={'b'}):
        def __init__(self):
            super().__init__()
            self.value_child_a = 0
            self.value_child_b = 0

        @striker.hooks.a
        def hook_child(self):
            self.value_child_a += 1

        @striker.hooks.b
        def hook_child_2(self):
            self.value_child_b += 1

        def run(self):
            self.hooks.run(type='a')
            self.hooks.run(type='a')
            self.hooks.run(type='b')

    p = Parent()
    p.run()
    assert p.value_parent == 2

    c = Child()
    c.run()
    assert c.value_parent == 2
    assert c.value_child_a == 2
    assert c.value_child_b == 1


def test_global_hook_type():
    """ Test that you can pass a global hook_type set """
    class Parent(striker.core.HookParent):
        @striker.hooks.a
        def hook_a(self):
            self.value_a += 1

        def hook_a_bis(self):
            self.value_a += 2

        def __init__(self):
            self.TYPES = set('a')
            self.hooks.check(self.TYPES)
            self.value_a = 0

            self.hooks.a[5](self.hook_a_bis)

        def run(self):
            self.hooks.run(type='a', index=1)
            assert self.value_a == 1

            self.hooks.run(type='a', index='5')
            assert self.value_a == 4
