# Script for automatic building, upload and install of a python module
# for the install we need to do it twice for obscure reasons, see https://github.com/pypa/pip/issues/7097
# must be called with ./modulecompiler 
# version 1.0
# Auteur : Guillaume GARRIGOS

$SETUP = 'setup.py'
if (Test-Path $SETUP)
{
    $NAME = python setup.py --name 
    $VERSION = python setup.py --version 
    $CONFIRM = Read-Host "You are going to build $NAME-$VERSION. 
Think about pushing your project before. 
Are you sure you want to proceed? (y/n)"
    if ($CONFIRM -eq 'y') 
    {
        python setup.py sdist install
        $CONFIRM = Read-Host "You are going to upload $NAME-$VERSION to PyPi. 
Proceed? (y/n)"
        if ($CONFIRM -eq 'y') 
        {
            python -m twine upload .\dist\$NAME-$VERSION*
            $CONFIRM = Read-Host "You are going to upgrade $NAME-$VERSION in your base environment. 
Proceed? (y/n)"
            if ($CONFIRM -eq 'y')
            {
                cd ..
                python -m pip install --upgrade $NAME
                python -m pip install --upgrade $NAME
                cd $NAME
            }
        }
    }
} 
else 
{"Error: There is no $SETUP file in this directory."}