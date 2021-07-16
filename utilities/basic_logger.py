import logging
import os


class BasicLogger(object):

    @staticmethod
    def setup_console_logging(config):
        """Sets up the logger"""
        logging_sv_path = os.path.join(config.folder['bl_log_sv'], "Training.log")
        logging.basicConfig(level=config.bl_console_logging_level,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=logging_sv_path,
                            filemode='w')

        #################################################################################################
        # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
        handler = logging.StreamHandler()
        handler.setLevel(config.bl_file_logging_level)
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logging.getLogger('').addHandler(handler)
        return logging
