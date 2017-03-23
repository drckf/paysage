import os, sys, pydoc, paysage
from collections import OrderedDict

module_header = "# {} documentation\n"
class_header = "## class {}"
function_header = "### {}"


def walk_through_package(package):
    output = OrderedDict()

    modules = pydoc.inspect.getmembers(package, pydoc.inspect.ismodule)

    for mod in modules:
        module_name, reference = mod
        output[module_name] = getmodule(module_name, reference)

    return output


def getmodule(module_name, reference):
    output = [module_name]

    if reference.__doc__:
        output.append(reference.__doc__)

    output.extend(getclasses(reference))
    funcs = getfunctions(reference)
    if funcs:
        output.extend(["## functions\n"])
        output.extend(funcs)

    return "\n".join((str(x) for x in output))


def getclasses(item):
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
            output.append(pydoc.inspect.getdoc(reference))
            # Get the methods
            output.extend(getmethods(reference))
            # Recurse into any methods
            output.extend(getclasses(reference))
            output.append('\n')

    return output


def getmethods(item):
    output = list()

    methods = pydoc.inspect.getmembers(item, pydoc.inspect.isfunction)

    for func in methods:

        func_name, reference = func

        if func_name.startswith('_') and func_name != '__init__':
            continue

        output.append(function_header.format(func_name.replace('_', '\\_')))

        # Get the signature
        output.append ('```py\n')
        output.append('def %s%s\n' % (
            func[0],
            pydoc.inspect.formatargspec(
                *pydoc.inspect.getfullargspec(reference)
            )))
        output.append ('```\n')

        # get the docstring
        if pydoc.inspect.getdoc(func[1]):
            output.append('\n')
            output.append(pydoc.inspect.getdoc(reference))

        output.append('\n')

    return output


def getfunctions(item):
    output = list()
    #print item
    for func in pydoc.inspect.getmembers(item, pydoc.inspect.isfunction):

        if func[0].startswith('_') and func[0] != '__init__':
            continue

        output.append(function_header.format(func[0].replace('_', '\\_')))

        # Get the signature
        output.append ('```py\n')
        output.append('def %s%s\n' % (
            func[0],
            pydoc.inspect.formatargspec(
                *pydoc.inspect.getfullargspec(func[1])
            )))
        output.append ('```\n')

        # get the docstring
        if pydoc.inspect.getdoc(func[1]):
            output.append('\n')
            output.append(pydoc.inspect.getdoc(func[1]))

        output.append('\n')
    return output
