def get_version():
	if get_version._version is None:
		from pkg_resources import get_distribution, DistributionNotFound

		try:
			get_version._version = get_distribution('hcipy').version
		except DistributionNotFound:
			# package is not installed
			pass

	return get_version._version

get_version._version = None