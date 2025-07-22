project('FSU_TEST', 'cpp',
        version : '1.0.0',
        default_options : ['warning_level=3', 'cpp_std=c++17'])

FSU_TEST = executable('FSU_TEST', 'main.cpp', install : true)

app_res_dir = meson.current_source_dir()
build_app_res_dir = meson.current_build_dir()
run_command(['cp', app_res_dir/'weights.bin', build_app_res_dir], check: true)

test('test', FSU_TEST)