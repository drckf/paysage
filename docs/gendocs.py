import pydoc
from collections import OrderedDict
import re
import os

module_header = "# Documentation for {0} ({1}.py)\n"
class_header = "## class {}"
function_header = "### {}"

def maybe_a(a, b, func):
    """
    Compute func(a, b) when a could be None.

    Args:
        a (any; maybe None)
        b (any)
        func (callable)

    Returns:
        func(a, b) or b if a is None

    """
    if a is not None:
        return func(a, b)
    return b

def write_into(addendum, out):
    """
    Writs addendum into out and returns a copy.

    Args:
        addendum (OrderedDict): dictionary to write into out
        out (OrderedDict): dictionary to write into

    Returns:
        out (OrderedDict): addendum written into out
    """
    for a in addendum:
        out[a] = addendum[a]
    return out

def save_doc_hierarchy(doc_od, basedir):
    """
    Saves the document hierarchy with a folder for each package
    and each module in each package having a single .md file.

    Args
        doc_od (OrderedDict): tree of of ordered dictionaries of docstrings

    Returns:
        None
    """
    for od in doc_od:
        if isinstance(doc_od[od], OrderedDict):
            subdir = os.path.join(basedir, od)
            try:
                os.mkdir(subdir)
            except OSError:
                pass

            save_doc_hierarchy(doc_od[od], subdir)
        else:
            savedoc(doc_od[od], od, basedir)

def savedoc(docstring, name, basedir):
    """
    Save the documentation string to a file named 'name' at basedir

    Args:
        docstring (str)
        name (str): filename
        basedir (os.path): base directory for *.md file

    Returns:
        None

    """
    path = os.path.join(basedir, name + '.md')
    with open(path, 'w') as output:
        output.write(docstring)


def format_linebreaks(string):
    """
    Replace '\n' linebreaks with '<br />' linebreaks for markdown:

    Args:
        string: A string

    Returns:
        string: A string with markdown style linebreaks.

    """
    return string.replace("\n", "<br />")


def format_indentation(string):
    """
    Replace indentation (4 spaces) with four HTML spaces

    Args:
        string: A string.

    Returns:
        string: A string.

    """
    return string.replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")

def document_modules(modules):
    """
    Get the documentation for each of the modules in a list of modules:

    Args:
        modules: A list of modules.

    Returns:
        output: A dictionary with documentation strings for each module.

    """
    output = OrderedDict()
    for mod in modules:
        as_name, reference = mod
        module_name = re.split('\.', reference.__name__)[-1]
        output[module_name] = getmodule(module_name, reference)
    return output

def walk_through_package_recursive(package):
    """
    Get the documentation for each of the modules in the package:

    Args:
        package: An imported python package.

    Returns:
        output: A tree of ordered dictionaries with leaves the
                documentation strings of submodules.

    """
    # get basedirectort of package
    basedir = os.path.dirname(package.__spec__.origin)

    # filters based on whether the module was imported from __init__.py or
    # if it came from elsewhere
    def from_init(mod):
        return maybe_a(mod.__file__,
                       False, lambda file,b: file.find('__init__.py')>-1)

    # filters based on whether the module is a python module object and
    # whether it is a submodule of the given package
    def is_local_sub_module(mod):
        return pydoc.inspect.ismodule(mod) and \
               maybe_a(mod.__spec__.origin, False,
                       lambda origin,b: origin.find(basedir) > -1)

    sub_packages = \
        pydoc.inspect.getmembers(package,
                                 lambda x: is_local_sub_module(x) and from_init(x))

    # For subpackages add another level to the tree
    output = OrderedDict()
    for sub in sub_packages:
        output[sub[0]] = walk_through_package_recursive(sub[1])

    # For proper submodules add a leaf
    modules = \
        pydoc.inspect.getmembers(package,
                                 lambda x: is_local_sub_module(x) and not from_init(x))
    output = write_into(document_modules(modules), output)
    return output


def getmodule(module_name, reference):
    """
    Get the documentation strings for a module by walking through the classes
    and functions.

    Args:
        module_name: The name of the module:
        reference: The module.

    Returns:
        str: The documentation string for the module.

    """
    output = [module_header.format(module_name.title(), module_name)]

    if reference.__doc__:
        output.append(reference.__doc__)

    output.extend(getclasses(reference))
    funcs = getfunctions(reference)
    if funcs:
        output.extend(["## functions\n"])
        output.extend(funcs)

    return "\n".join((str(x) for x in output))


def getclasses(item):
    """
    Get the documentation strings for each class in a module.

    Args:
        item: The module.

    Returns:
        list: Documentation strings for each class and method in the module.

    """
    output = list()
    # we use some class aliases and don't want to be redundant
    # keep a list of unique classes (by reference, not name)
    base_references = list()

    # get all of the classes in the module and sort them so that
    # classes with longer names are on the top of the list
    classes = pydoc.inspect.getmembers(item, pydoc.inspect.isclass)
    classes.sort(key=lambda x: len(x[0]), reverse=True)

    for cl in classes:

        class_name, reference = cl

        # only keep non-redundant classes
        if reference in base_references:
            continue
        else:
            base_references.append(reference)

        if  not class_name.startswith("_"):
            # Consider anything that starts with _ private
            # and don't document it
            output.append( class_header.format(class_name))
            # Get the docstring
            docstring = pydoc.inspect.getdoc(reference)
            if docstring:
                output.append(format_linebreaks(docstring))
            # Get the methods
            output.extend(getfunctions(reference))
            # Recurse into any methods
            output.extend(getclasses(reference))
            output.append('\n')

    return output


def getfunctions(item):
    """
    Get the documentation strings for each function in the item
    (a module or class).

    Args:
        item: A module or class.

    Returns:
        list: Documentation strings for each function.

    """

    output = list()
    # filters based on whether the module function is coming from is local
    def is_local_func(mod):
        return pydoc.inspect.isfunction(mod) and \
               mod.__module__.find('paysage') > -1

    methods = pydoc.inspect.getmembers(item, is_local_func)

    for func in methods:

        func_name, reference = func

        if func_name.startswith('_') and func_name != '__init__':
            continue

        output.append(function_header.format(func_name.replace('_', '\\_')))

        # get argspec
        argspec = pydoc.inspect.getfullargspec(reference)
        arg_text = pydoc.inspect.formatargspec(*argspec)

        _re_stripid = re.compile(r' at 0x[0-9a-f]{6,16}(>+)', re.IGNORECASE)
        def stripid(text):
            """
            Strips off object ids
            """
            return _re_stripid.sub(r'\1', text)

        # Get the signature
        output.append ('```py\n')
        output.append('def %s%s\n' % (
            func_name,
            stripid(arg_text)
            ))
        output.append ('```\n')

        # get the docstring
        docstring = pydoc.inspect.getdoc(reference)
        if docstring:
            output.append('\n')
            output.append(format_indentation(
                          format_linebreaks(docstring)
                          ))

        output.append('\n')

    return output


if __name__ == "__main__":
    import paysage
    doc_hierarchy = walk_through_package_recursive(paysage)
    save_doc_hierarchy(doc_hierarchy, os.path.dirname(__file__))
