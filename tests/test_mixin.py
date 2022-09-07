from typing import Protocol

import pytest
import striker
striker.core.HookParent.__hook_check__ = 'raise'


def test_api():
    """ Test general mixin API """
    class ParentProtocol(Protocol):
        value: int
        cd: int

    class CustomMixin(striker.core.Mixin, hook_types={'a'}, parent_protocol=ParentProtocol):
        def __init__(self):
            self.value = 0

        @striker.hooks.a
        def hook_a(self):
            self.value += 1

        def run(self):
            self.parent.value += self.value * self.parent.cd

    class Parent(striker.core.MixinParent):
        custom_mixin = CustomMixin()

        def __init__(self):
            self.value = 0
            self.mixins.check()

        def __getattr__(self, name):
            if name == 'ab':
                return None
            else:
                raise AttributeError(f'{name} attribute does not exist')

        @property
        def cd(self):
            return 2

        def run(self):
            self.mixins.run(type='a')
            self.custom_mixin.run()
            assert self.value == 2

    p = Parent()
    p.run()


def test_protocol_fail():
    """ Test that the code raises an error when protocol is not matched. """
    class ParentProtocol(Protocol):
        doesnotexist: float

    class CustomMixin(striker.core.Mixin, parent_protocol=ParentProtocol):
        pass

    class Parent(striker.core.MixinParent):
        custom_mixin = CustomMixin()

        def __init__(self):
            self.mixins.check()

    with pytest.raises(TypeError):
        Parent()


def test_multiple_parents():
    """ Test that plugins work correctly when multiple PluginParent instances are created """
    class CustomMixin(striker.core.Mixin, hook_types={'a'}):
        def __init__(self):
            self.value = 0

        @striker.hooks.a
        def hook_a(self):
            self.value += 1

        def run(self):
            self.parent.value += self.value * self.parent.cd

    class Parent(striker.core.MixinParent):
        custom_mixin = CustomMixin()

        def __init__(self):
            self.value = 0

        @property
        def cd(self):
            return 2

        def run(self):
            self.mixins.run(type='a')
            self.custom_mixin.run()
            assert self.value == 2

    p1 = Parent()
    p2 = Parent()

    p1.run()
    assert p1.value == 2
    assert p2.value == 0

    p2.run()
    assert p1.value == 2
    assert p2.value == 2

    p3 = Parent()
    assert p3.value == 0

    p3.run()
    assert p1.value == 2
    assert p2.value == 2
    assert p3.value == 2


def test_inheritance_mixin():
    """ Test that plugins work correctly when it is inherited """
    class ParentMixin(striker.core.Mixin, hook_types={'a'}):
        def __init__(self):
            self.value = 0

        @striker.hooks.a
        def hook_a(self):
            self.value += 1

        def run(self):
            self.parent.value_a += self.value * self.parent.double

    class ChildMixin(ParentMixin, hook_types=['b']):
        @striker.hooks.b
        def hook_b(self):
            self.parent.value_b += 1

        def run(self):
            self.parent.value_a += self.value * self.parent.double + self.parent.value_b

    class Parent(striker.core.MixinParent):
        parent_mixin = ParentMixin()
        child_mixin = ChildMixin()

        def __init__(self):
            self.value_a = 0
            self.value_b = 0

        @property
        def double(self):
            return 2

        def run(self):
            self.mixins.run(type='a')
            self.parent_mixin.run()
            assert self.value_a == 2
            assert self.value_b == 0

            self.mixins.run(type='b')
            assert self.value_a == 2
            assert self.value_b == 1

            self.child_mixin.run()
            assert self.value_a == 5
            assert self.value_b == 1

    p = Parent()
    p.run()
