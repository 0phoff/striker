from __future__ import annotations
from typing import Any
from ._manager import MixinManager

__all__ = ['MixinParent']


class MixinParent:
    """
    Any class that wants to use mixins needs to subclass :class:`~striker.core.mixin.MixinParent`.

    Attributes:
        mixins (MixinManager): Manager to run manage mixins bound to that specific class instance

    Example:
        >>> class Parent(striker.core.mixin.MixinParent):
        ...     trainloop = CustomLoopMixin()
        ...
        ...     def __init__(self):
        ...         # Recommended: Check thath there are only valid hooks
        ...         self.mixins.check()
        ...
        ...     def run(self):
        ...         # Each mixin should document what hooks they use
        ...         self.mixin.run(type='a')
        ...         # Assuming the mixin has a run method
        ...         self.trainloop.run()
    """
    mixins: MixinManager

    def __new__(cls, *args: Any, **kwargs: Any) -> MixinParent:
        obj = super().__new__(cls)
        obj.mixins = MixinManager(obj)
        return obj
