import sys


def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print('{} - {}'.format(exc_type, exc_tb.tb_lineno))
            return None

    return safe_f
