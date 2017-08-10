import pydoc
from collections import OrderedDict

module_header = "# Documentation for {0} ({1}.py)\n"
class_header = "## class {}"
function_header = "### {}"


def savedocs(docs):
    """
    Save the documentation to a series of markdown files for mkdocs.

    Notes:
        Outputs files.

    Args:
        docs: A dictionary of strings:

    Returns:
        None

    """
    for key in docs:
        with open(key + '.md', 'w') as output:
            output.write(docs[key])


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
    Replace indentation (4 spaces) with '\t'

    Args:
        string: A string.

    Returns:
        string: A string.

    """
    return string.replace("    ", " ~ ")


def walk_through_package(package):
    """
    Get the documentation for each of the modules in the package:

    Args:
        package: An imported python package.

    Returns:
        output: A dictionary with documentation strings for each module.

    """
    output = OrderedDict()

    modules = pydoc.inspect.getmembers(package, pydoc.inspect.ismodule)

    for mod in modules:
        module_name, reference = mod
        output[module_name] = getmodule(module_name, reference)
    return output


def getmodule(module_name, reference):
    """
    Get the documentation strings for a module by walking through the classes
    and functions.

    Args:
        module_name: The name of the module:
        reference: The module.

    Returns:
        str: The documentation string for the modulee.

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

    methods = pydoc.inspect.getmembers(item, pydoc.inspect.isfunction)

    for func in methods:

        func_name, reference = func

        if func_name.startswith('_') and func_name != '__init__':
            continue

        output.append(function_header.format(func_name.replace('_', '\\_')))

        # Get the signature
        output.append ('```py\n')
        output.append('def %s%s\n' % (
            func_name,
            pydoc.inspect.formatargspec(
                *pydoc.inspect.getfullargspec(reference)
            )))
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
    # TODO: define this to run recursively
    savedocs(walk_through_package(paysage))
    savedocs(walk_through_package(paysage.models))
    savedocs(walk_through_package(paysage.layers))
