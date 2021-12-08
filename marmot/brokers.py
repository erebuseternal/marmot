import geopandas


class ShapeFileBroker(object):
    def __init__(self, **config):
        self.file_path = config['source_path']
        self.file_paths = [self.file_path]
        self.geometry_key = config.get('geometry_key', 'geometry')

    def _build_key(self, index, file_path):
        return f'{index}@{file_path}'

    def _decompose_key(self, key):
        index, file_path = key.split('@')
        return int(index), file_path

    def read_and_build_keys(self):
        for file_path in self.file_paths:
            shapes = geopandas.read_file(file_path)
            for index, row in shapes.iterrows():
                yield self._build_key(index, file_path), row[self.geometry_key]

    def read(self, key):
        index, file_path = self._decompose_key(key)
        return geopandas.read_file(file_path).iloc[index][self.geometry_key]
   