def list_all_files_with_extension(path, extension):
	import os
	files = os.listdir(path)
	files = filter(lambda name: os.path.splitext(name)[1] == extension, files)
	return files