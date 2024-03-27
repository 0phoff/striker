from typing import Protocol

import pytest

import striker

striker.core.HookParent.__hook_check__ = 'raise'


def test_api():
    """ Test general plugin API """
    class ParentProtocol(Protocol):
        value: int

    class CustomPlugin(striker.core.Plugin, hook_types={'a', 'b'}, parent_protocol=ParentProtocol):
        def __init__(self):
            self.value = 0

        @striker.hooks.a
        def hook_a(self):
            self.value += 1

        @striker.hooks.b
        def hook_b(self):
            self.parent.value += 1

    class Parent(striker.core.PluginParent):
        plugins = [CustomPlugin()]

        def __init__(self):
            self.value = 0
            self.plugins.check()

        def run(self):
            self.plugins.run(type='a')
            assert self.plugins['customplugin'].value == 1

            self.plugins.run(type='b')
            assert self.value == 1

    p = Parent()
    p.run()

    # Plugin instance on class should still be untouched
    assert Parent.plugins[0].value == 0


def test_protocol_fail():
    """ Test that the code raises an error when protocol is not matched. """
    class ParentProtocol(Protocol):
        doesnotexist: float

    class CustomPlugin(striker.core.Plugin, parent_protocol=ParentProtocol):
        pass

    class Parent(striker.core.PluginParent):
        plugins = [CustomPlugin()]

        def __init__(self):
            self.plugins.check()

    with pytest.raises(TypeError):
        Parent()


def test_disable():
    """ Test that disabling a plugin works correctly """
    class CustomPlugin(striker.core.Plugin, hook_types=['a']):
        @striker.hooks.a
        def hook(self):
            self.parent.value += 1

    class Parent(striker.core.PluginParent):
        plugins = [CustomPlugin()]

        def __init__(self):
            self.plugins.check()
            self.value = 0

        def run(self):
            self.plugins.run(type='a')
            assert self.value == 1

            self.plugins['customplugin'].enabled = False
            assert not self.plugins['customplugin'].enabled
            self.plugins.run(type='a')
            assert self.value == 1

            self.plugins['customplugin'].enabled = True
            self.plugins.run(type='a')
            assert self.value == 2

    p = Parent()
    p.run()


def test_multiple_parents():
    """ Test that plugins work correctly when multiple PluginParent instances are created """
    class CustomPlugin(striker.core.Plugin, hook_types=['a']):
        @striker.hooks.a
        def hook(self):
            self.parent.value += 1

    class Parent(striker.core.PluginParent):
        plugins = [CustomPlugin()]

        def __init__(self):
            self.plugins.check()
            self.value = 0

        def run(self):
            self.plugins.run(type='a')
            self.plugins.run(type='a')
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


def test_inheritance_parent():
    """ Test that plugins work correctly when a PluginParent is inherited """
    class CustomPluginA(striker.core.Plugin, hook_types=['a']):
        @striker.hooks.a
        def hook(self):
            self.parent.value_a += 1

    class CustomPluginB(striker.core.Plugin, hook_types=['b']):
        @striker.hooks.b
        def hook(self):
            self.parent.value_b += 1

    class Parent(striker.core.PluginParent):
        plugins = [CustomPluginA()]

        def __init__(self):
            self.plugins.check()
            self.value_a = 0

        def run(self):
            self.plugins.run(type='a')
            self.plugins.run(type='a')
            assert self.value_a == 2

    class Child(Parent):
        plugins = [CustomPluginB()]

        def __init__(self):
            super().__init__()
            self.value_b = 0

        def run(self):
            super().run()
            assert self.value_b == 0
            self.plugins.run(type='b')
            assert self.value_a == 2
            assert self.value_b == 1

    p = Parent()
    p.run()

    c = Child()
    c.run()


def test_inheritance_plugin():
    """ Test that plugins work correctly when it is inherited """
    class ParentPlugin(striker.core.Plugin, hook_types=['a']):
        @striker.hooks.a
        def hook_a(self):
            self.parent.value_a += 1

    class ChildPlugin(ParentPlugin, hook_types=['b']):
        @striker.hooks.b
        def hook_b(self):
            self.parent.value_b += 1

    class Parent(striker.core.PluginParent):
        plugins = [ParentPlugin(), ChildPlugin()]

        def __init__(self):
            self.value_a = 0
            self.value_b = 0

        def run(self):
            self.plugins.run(type='a')
            assert self.value_a == 2
            assert self.value_b == 0

            self.plugins.run(type='b')
            assert self.value_a == 2
            assert self.value_b == 1

    p = Parent()
    p.run()
