#
#   filename: base_processor.py
#       A class to read data from file, parse data to dict structure and save data to MongoDB
#   auther: Maureen Yang
#   data: 2018/07/18
#

class base_processor():
    """
        this is description of base processor
    """
    __type__ = 'Base Processor'
    __data_dict__ = None

    def __init__(self):
        pass

    def read(self):
        print(self.__type__,' read function')

    def get_dict(self):
        print(self.__type__,' get dict function')

    def save2DB(self):
        print(self.__type__,' save to DB function')
