import os, sys, pydoc, paysage

module_header = "# {} documentation\n"
class_header = "## class {}"
function_header = "### {}"


def walk_through_package(package):
    output = list()
    for mod in pydoc.inspect.getmembers(package, pydoc.inspect.ismodule):
        output.append(getmarkdown(mod[1]))
    return output

def getmarkdown(module):
    output = [ module_header.format(module.__name__) ]

    if module.__doc__:
        output.append(module.__doc__)

    output.extend(getclasses(module))
    funcs = getfunctions(module)
    if funcs:
        output.extend(["## functions\n"])
        output.extend(funcs)

    return "\n".join((str(x) for x in output))

def getclasses(item):
    output = list()
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

        if func[0].startswith('_') and func[0] != '__init__':
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

def generatedocs(module):
    try:
        sys.path.append(os.getcwd())
        # Attempt import
        mod = pydoc.safeimport(module)
        if mod is None:
           print("Module not found")

        # Module imported correctly, let's create the docs
        return getmarkdown(mod)
    except pydoc.ErrorDuringImport as e:
        print("Error while trying to import " + module)
