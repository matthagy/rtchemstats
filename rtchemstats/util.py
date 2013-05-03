#  Copyright (C) 2012 Matt Hagy <hagy@gatech.edu>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


from functools import wraps

def cached_property(name_or_func, cache_name=None):
    '''Wrapper to create a cached readonly property for a class.
    '''
    if isinstance(name_or_func, basestring):
        return lambda func: cached_property(func, cache_name=name_or_func)

    func = name_or_func
    assert callable(func)

    if cache_name is None:
        cache_name = '_' + func.func_name
    assert isinstance(cache_name, basestring)

    @property
    @wraps(func)
    def wrapper(self):
        try:
            return getattr(self, cache_name)
        except AttributeError:
            value = func(self)
            setattr(self, cache_name, value)
            return value
    return wrapper
