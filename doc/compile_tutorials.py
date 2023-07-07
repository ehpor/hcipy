import os
import shutil
import time
import sys

from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbconvert.exporters import RSTExporter
from nbconvert.writers import FilesWriter
import nbformat

from PIL import Image, ImageChops

def compile_tutorial(tutorial_name, force_recompile=False):
    print('- Tutorial "' + tutorial_name + '"')

    notebook_path = 'tutorial_notebooks/' + tutorial_name + '/' + tutorial_name + '.ipynb'
    export_path = 'tutorials/' + tutorial_name + '/' + tutorial_name
    thumb_dest = os.path.dirname(export_path) + '/thumb.png'

    if not os.path.exists(os.path.dirname(export_path)):
        os.makedirs(os.path.dirname(export_path))

    # Read in notebook
    print('  Reading notebook...')
    notebook = nbformat.read(notebook_path, 4)

    # Scrape title, description and thumbnail
    first_cell = notebook.cells[0]

    title = first_cell.source.splitlines()[0]
    if '#' in title:
        title = title.replace('#', '').strip()

    description = ''
    for line in first_cell.source.splitlines()[1:]:
        if line.strip():
            description = line.strip()
            break

    if not description:
        print('  Description could not be found in the notebook.')

    if 'thumbnail_figure_index' in notebook.metadata:
        thumbnail_figure_index = notebook.metadata['thumbnail_figure_index']
    else:
        thumbnail_figure_index = -1

    if 'level' in notebook.metadata:
        level = notebook.metadata['level'].capitalize()
    elif 'difficulty' in notebook.metadata:
        level = notebook.metadata['difficulty'].capitalize()
    else:
        level = 'Unknown'

    # Check if the tutorial was already executed.
    if os.path.exists(export_path + '.rst'):
        if os.path.getmtime(export_path + '.rst') > os.path.getmtime(notebook_path):
            if force_recompile:
                print('  Already compiled. Recompiling anyway...')
            else:
                print('  Already compiled. Skipping compilation...')
                return title, level, description, thumb_dest.split('/', 1)[-1], False

    # Execute notebook if not already executed
    already_executed = any(c.get('outputs') or c.get('execution_count') for c in notebook.cells if c.cell_type == 'code')

    resources = {'metadata': {'path': os.path.dirname(notebook_path)}}

    with_errors = False

    if not already_executed:
        print('  Executing', end='')
        start = time.time()

        additional_cell_1 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": r"%matplotlib inline" + '\n' +
                r"%config InlineBackend.print_figure_kwargs = {'bbox_inches': None}"
            }

        additional_cell_2 = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "import matplotlib as mpl\nmpl.rcParams['figure.figsize'] = (8, 6)\nmpl.rcParams['figure.dpi'] = 150\nmpl.rcParams['savefig.dpi'] = 150"
            }

        notebook.cells.insert(1, nbformat.from_dict(additional_cell_1))
        notebook.cells.insert(2, nbformat.from_dict(additional_cell_2))

        client = NotebookClient(nb=notebook, resources=resources, timeout=585, kernel_name='python3')

        try:
            with client.setup_kernel():
                for i, cell in enumerate(notebook.cells):
                    print('.', end='')

                    client.execute_cell(cell, i)

            client.set_widgets_metadata()
        except CellExecutionError as err:
            print('  Error while processing notebook:')
            print('  ', err)

            with_errors = True

        print('')

        notebook.cells.pop(2)
        notebook.cells.pop(1)

        end = time.time()

        time_taken = end - start
        if time_taken > 60:
            print('  Execution took %dm%02ds.' % (time_taken / 60, time_taken % 60))
        else:
            print('  Execution took %ds.' % time_taken)
    else:
        print('  Notebook was already executed.')

    print('  Rendering tutorial...')
    exporter = RSTExporter()
    output, resources = exporter.from_notebook_node(notebook, resources)

    writer = FilesWriter(build_directory=os.path.dirname(export_path))
    writer.write(output, resources, notebook_name=os.path.basename(export_path))

    pictures = sorted(resources['outputs'], key=output.find)

    try:
        thumbnail_source = pictures[thumbnail_figure_index]

        # Read in thumbnail source image
        img = Image.open(os.path.dirname(export_path) + '/' + thumbnail_source)

        # Trim whitespace
        bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff)
        bbox = diff.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Resize image to have a width of 400px
        img.thumbnail([400, 1000])

        # Save thumbnail
        img.save(thumb_dest)
    except:
        shutil.copyfile('_static/no_thumb.png', thumb_dest)

    print('  Done!')

    return title, level, description, thumb_dest.split('/', 1)[-1], with_errors

index_preamble = '''
Tutorials
=========

These tutorials demonstrate the features of HCIPy in the context of a standard workflow. Tutorials are separated in three categories, depending on the required level of familiarity with HCIPy.
'''

entry_template = '''
.. only:: html

    .. container:: tutorial_item

        :doc:`{title} <{name}/{name}>`

        .. container:: tutorial_row

            .. container:: tutorial_thumbnail

                .. figure:: {thumbnail_file}

            .. container:: tutorial_description

                **Level:** {level}

                **Description:** {description}
'''

beginner_preamble = '''
Beginner
--------

These tutorials provide an introduction to the basic parts of HCIPy. New users should read these tutorials to get started with HCIPy.
'''

intermediate_preamble = '''
Intermediate
------------

These tutorials show the main functionality using the built-in classes of HCIPy. These tutorials focus on one aspect of high-contrast imaging.
'''

advanced_preamble = '''
Advanced
--------
These tutorials show how to use HCIPy for your own research. This includes extending HCIPy with your own optical elements and advanced use cases.
'''

expert_preamble = '''
Expert
------

These tutorials provide examples from actual published research that made heavy use of HCIPy for their optical propagations.
'''

unknown_preamble = '''
Unknown
-------

These tutorials do not have their level of difficulty rated.
'''

level_preambles = [beginner_preamble, intermediate_preamble, advanced_preamble, expert_preamble, unknown_preamble]

def compile_all_tutorials():
    print('Compiling all tutorials...')

    tutorials = {}
    tutorial_names = sorted(os.listdir('tutorial_notebooks/'))
    tutorial_names = [name for name in tutorial_names if 'checkpoint' not in name]

    with_errors = False

    for name in tutorial_names:
        tutorials[name] = compile_tutorial(name)

        with_errors = with_errors or tutorials[name][-1]

    # Sort by level
    levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert', 'Unknown']

    tutorial_names_by_level = [[] for i in range(len(levels))]
    for name in tutorial_names:
        tutorial_names_by_level[levels.index(tutorials[name][1])].append(name)

    with open('tutorials/index.rst', 'w') as f:
        f.write(index_preamble)

        for preamble, names in zip(level_preambles, tutorial_names_by_level):
            # Don't write this level if there are no tutorials in it.
            if len(names) == 0:
                continue

            f.write(preamble)

            # Write toctree
            f.write('\n.. toctree::\n    :maxdepth: 1\n    :hidden:\n\n')
            for name in names:
                f.write('    ' + name + '/' + name + '\n')
            f.write('\n\n')

            # Write list
            for name in names:
                title, level, desc, thumb, _ = tutorials[name]
                f.write(entry_template.format(thumbnail_file=thumb, title=title, level=level, description=desc, name=name))

    return with_errors

if __name__ == '__main__':
    # Compile all tutorials
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    import sys
    sys.path.insert(0, '.')

    with_errors = compile_all_tutorials()

    if with_errors:
        sys.exit(1)
