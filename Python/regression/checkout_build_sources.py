from __future__ import print_function
import argparse
import os
import shutil
import subprocess as sp
import sys

import yaml

def run_cmd(args, cwd='.', raise_err=True):
    if isinstance(args, str):
        args = args.split()
    print("RUN CMD", args, file=sys.stderr)
    proc =  sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT, cwd=cwd, env=os.environ)
    lines = []
    while proc.poll() is None:
        line = proc.stdout.readline()
        print(line, end='', file=sys.stderr)
        lines.append(line)
    new_lines = proc.stdout.readlines()
    print(''.join(new_lines), file=sys.stderr)
    if proc.poll() and raise_err:
        raise ValueError("Subprocess failed {}".format(proc.poll()))
    return lines

_d = os.path.dirname
REGRESSION_CONFIG = os.path.join(_d(_d(_d(os.path.abspath(__file__)))), '.regression.yml')
REGRESSION_CONFIG = yaml.load(open(REGRESSION_CONFIG))
REQUIRED = set(('compare_taxcalc_version',
                'compare_ogusa_version',
                'install_taxcalc_version',
                'diff',
                'numpy_version'))
if not set(REGRESSION_CONFIG) >= REQUIRED:
    raise ValueError('.regression.yml at top level of repo needs to define: '.format(REQUIRED - set(REGRESSION_CONFIG)))

OGUSA_ENV_PATH = os.path.join(os.environ['WORKSPACE'], 'ogusa_env')

# Should be set by build script:
MINICONDA_ROOT = os.environ['MINICONDA_ROOT']
MINICONDA_ENV = os.environ['MINICONDA_ENV']
CONDA_ROOT = os.path.join(MINICONDA_ROOT, 'bin', 'conda')
CONDA = os.path.join(MINICONDA_ENV, 'bin', 'conda')
PYTHON = os.path.join(MINICONDA_ENV, 'bin', 'python')

def cli():
    parser = argparse.ArgumentParser(description='Get install OG-USA branch')
    parser.add_argument('action', choices=['get_miniconda',
                                           'make_ogusa_env',
                                           'customize_ogusa_env'])
    parser.add_argument('ogusabranch')
    return parser.parse_args()



def get_miniconda(args):
    print("GET MINICONDA")
    numpy_vers = REGRESSION_CONFIG['numpy_version']
    install_ogusa_version = args.ogusabranch
    install_taxcalc_version = REGRESSION_CONFIG['install_taxcalc_version']
    compare_ogusa_version = REGRESSION_CONFIG['compare_ogusa_version']
    compare_taxcalc_version = REGRESSION_CONFIG['compare_taxcalc_version']
    run_cmd('wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh')
    run_cmd('bash miniconda.sh -b -p {}'.format(MINICONDA_ROOT))
    print("GET MINICONDA OK")

def make_ogusa_env(args):
    print('MAKE OGUSA ENV')
    run_cmd('{} config --set always_yes yes --set changeps1 no'.format(CONDA_ROOT))
    run_cmd('{} update conda -n root'.format(CONDA_ROOT))
    lines = ' '.join(run_cmd('{} env list'.format(CONDA_ROOT))).lower()
    if 'ogusa_env' in lines:
        run_cmd('{} env remove --path {}'.format(CONDA_ROOT, MINICONDA_ENV))
    run_cmd('{} install nomkl'.format(CONDA_ROOT))
    run_cmd('{} create -p {} python=2.7 yaml'.format(CONDA_ROOT, MINICONDA_ENV))
    print('MAKE OGUSA ENV OK')

def customize_ogusa_env(args):
    print('CUSTOMIZE OGUSA ENV')

    numpy_vers = REGRESSION_CONFIG['numpy_version']
    install_ogusa_version = args.ogusabranch
    install_taxcalc_version = REGRESSION_CONFIG['install_taxcalc_version']
    compare_ogusa_version = REGRESSION_CONFIG['compare_ogusa_version']
    compare_taxcalc_version = REGRESSION_CONFIG['compare_taxcalc_version']
    run_cmd('{} install --force -c ospc openblas pytest toolz scipy numpy={} pandas=0.18.1 matplotlib'.format(CONDA, numpy_vers))
    run_cmd('{} remove mkl mkl-service'.format(CONDA), raise_err=False)
    run_cmd('{} install -c ospc taxcalc={} --force'.format(CONDA, install_taxcalc_version))
    run_cmd('git fetch --all')
    run_cmd('git checkout regression')
    regression_tmp = os.path.join('..', 'regression')
    if os.path.exists(regression_tmp):
        shutil.rmtree(regression_tmp)
    src = os.path.join('Python', 'regression')
    shutil.copytree(src, regression_tmp)
    run_cmd('git checkout {}'.format(install_ogusa_version))
    if not os.path.exists(src):
        shutil.copytree(regression_tmp, src)
    run_cmd('{} setup.py install'.format(PYTHON))
    puf_choices = (os.path.join('..', '..', 'puf.csv'),
                   os.path.join('Python', 'regression', 'puf.csv'),
                   os.path.join('/home', 'ubuntu', 'deploy', 'puf.csv'))

    puf = [puf for puf in puf_choices if os.path.exists(puf)]
    if not puf:
        raise ValueError('Could not find puf.csv at {}'.format(puf_choices))
    puf = puf[0]
    shutil.copy(puf, os.path.join('Python', 'regression', 'puf.csv'))
    print('CUSTOMIZE OGUSA ENV OK')
    return 0


if __name__ == "__main__":
    args = cli()
    if args.action == 'get_miniconda':
        get_miniconda(args)
    elif args.action == 'make_ogusa_env':
        make_ogusa_env(args)
    else:
        customize_ogusa_env(args)