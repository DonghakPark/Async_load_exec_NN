project('FSU_TEST', 'cpp',
        version : '1.0.0',
        default_options : ['warning_level=3', 'cpp_std=c++17'])

bs_thread_pool = [
        'main.cpp',
        'bs_thread_pool_manager.cpp'
]

FSU_TEST = executable('FSU_TEST',
                      bs_thread_pool,
                      include_directories : [include_directories('.')],
                      install : false)

#FSU_TEST = executable('FSU_TEST', 'main.cpp', install : false)
IO_TEST = executable('IO_TEST', 'io_test.cpp', install : false)