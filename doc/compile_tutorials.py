import os
import shutil

from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from nbconvert.exporters import RSTExporter
from nbconvert.writers import FilesWriter
import nbformat

from PIL import Image, ImageChops

def compile_tutorial(tutorial_name):
	print('Compiling tutorial ' + tutorial_name + '...')

	notebook_path = 'tutorial_notebooks/' + tutorial_name + '/' + tutorial_name + '.ipynb'
	export_path = 'tutorials/' + tutorial_name + '/' + tutorial_name

	if not os.path.exists(os.path.dirname(export_path)):
		os.makedirs(os.path.dirname(export_path))
	
	# Read in notebook
	notebook = nbformat.read(notebook_path, 4)

	# Scrape title, description and thumbnail
	first_cell = notebook.cells[0]

	title = first_cell.source.splitlines()[0]
	if '#' in title:
		title = title.replace('#', '').strip()
	
	description = first_cell.source.splitlines()[2].strip()
	
	if 'thumbnail_figure_index' in notebook.metadata:
		thumbnail_figure_index = notebook.metadata['thumbnail_figure_index']
	else:
		thumbnail_figure_index = -1

	# Execute notebook if not already executed
	already_executed = any(c.get('outputs') or c.get('execution_count') for c in notebook.cells if c.cell_type == 'code')

	resources = {}
	
	if not already_executed:
		ep = ExecutePreprocessor(timeout=120, kernel_name='python3')
		try:
			notebook, resources = ep.preprocess(notebook)
		except CellExecutionError as err:
			print('Error while processing notebook.')
			print(err)

	exporter = RSTExporter()
	output, resources = exporter.from_notebook_node(notebook, resources)

	writer = FilesWriter(build_directory=os.path.dirname(export_path))
	writer.write(output, resources, notebook_name=os.path.basename(export_path))

	# 400, 280
	pictures = sorted(resources['outputs'], key=output.find)
	thumb_dest = os.path.dirname(export_path) + '/thumb.png'

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

		# Resize image
		img.thumbnail([200, 150])

		# Save thumbnail
		img.save(thumb_dest)
	except:
		shutil.copyfile('_static/no_thumb.png', thumb_dest)

	return title, description, thumb_dest.split('/', 1)[-1]

index_preamble = '''
Tutorials
=========

These tutorials demonstrate the features of HCIPy in the context of a standard workflow.
'''

entry_template = '''
.. only:: html

    .. container:: tutorial_item

        :doc:`{title} <{name}/{name}>`

        .. container:: tutorial_thumbnail

            .. figure:: {thumbnail_file}

        .. container:: tutorial_description
			
            {description}
'''

def compile_all_tutorials():
	print('Compling all tutorials...')

	tutorials = {}
	tutorial_names = sorted(os.listdir('tutorial_notebooks/'))

	for name in tutorial_names:
		tutorials[name] = compile_tutorial(name)

	f = open('tutorials/index.rst', 'w')
	f.write(index_preamble)

	# Write toctree
	f.write('\n.. toctree::\n    :maxdepth: 1\n    :hidden:\n\n')
	for name in tutorial_names:
		f.write('    ' + name + '/' + name + '\n')
	f.write('\n\n')
	
	# Write list
	for name in tutorial_names:
		title, desc, thumb = tutorials[name]

		f.write(entry_template.format(thumbnail_file=thumb, title=title, description=desc, name=name))
	
	f.close()
