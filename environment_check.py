import os
import sys
import importlib
import platform

def check_module(module_name, min_version=None):
    """Check if a module is installed and meets minimum version requirements"""
    try:
        # Special case for OpenCV
        if module_name == 'opencv-python':
            try:
                import cv2
                version = cv2.__version__
                return {
                    'installed': True,
                    'version': version,
                    'meets_min_version': True if not min_version else version >= min_version
                }
            except ImportError:
                return {
                    'installed': False,
                    'version': None,
                    'meets_min_version': False
                }
        else:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', None)
        
        if version is None:
            # Try some common version attributes
            if hasattr(module, 'version'):
                version = getattr(module, 'version')
            elif hasattr(module, 'VERSION'):
                version = getattr(module, 'VERSION')
            
        result = {
            'installed': True,
            'version': version
        }
        
        if min_version and version:
            result['meets_min_version'] = version >= min_version
        else:
            result['meets_min_version'] = True
            
        return result
    except ImportError:
        return {
            'installed': False,
            'version': None,
            'meets_min_version': False
        }

def check_directory(path, create_if_missing=False):
    """Check if a directory exists and create it if specified"""
    result = {'exists': os.path.exists(path), 'is_dir': False, 'created': False}
    
    if result['exists']:
        result['is_dir'] = os.path.isdir(path)
    elif create_if_missing:
        try:
            os.makedirs(path, exist_ok=True)
            result['exists'] = True
            result['is_dir'] = True
            result['created'] = True
        except Exception as e:
            result['error'] = str(e)
            
    return result

def check_environment():
    """Check if the environment is properly set up for Musotion"""
    print("===== Musotion Environment Check =====")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    if python_version < "3.8":
        print("WARNING: Python 3.8 or higher is recommended")
    
    # Check operating system
    print(f"Operating system: {platform.system()} {platform.release()}")
    
    # Check required modules
    required_modules = [
        ('numpy', '1.19.0'),
        ('pandas', '1.0.0'),
        ('tensorflow', '2.0.0'),
        ('keras', '2.3.0'),
        ('pygame', '2.0.0'),
        ('opencv-python', '4.0.0'),
        ('imutils', '0.5.0')
    ]
    
    print("\n----- Required Packages -----")
    all_modules_installed = True
    
    for module_name, min_version in required_modules:
        result = check_module(module_name, min_version)
        
        if result['installed']:
            status = f"✓ {module_name} - Version: {result['version']}"
            if not result['meets_min_version'] and min_version:
                status += f" (WARNING: Minimum recommended version is {min_version})"
                all_modules_installed = False
        else:
            status = f"✗ {module_name} - Not installed"
            all_modules_installed = False
            
        print(status)
    
    # Check project directories
    print("\n----- Project Directories -----")
    directories = [
        ('music', True),
        ('output_model', False),
        ('emoji', True)
    ]
    
    for dir_name, create in directories:
        result = check_directory(dir_name, create)
        
        if result['exists'] and result['is_dir']:
            if result['created']:
                print(f"✓ {dir_name} - Created directory")
            else:
                print(f"✓ {dir_name} - Directory exists")
        else:
            print(f"✗ {dir_name} - Directory does not exist or isn't accessible")
            if 'error' in result:
                print(f"  Error: {result['error']}")
    
    # Check if haarcascade file exists
    print("\n----- Required Files -----")
    cascade_file = 'haarcascade_frontalface_alt2.xml'
    
    if os.path.exists(cascade_file):
        print(f"✓ {cascade_file} - File exists")
    else:
        print(f"✗ {cascade_file} - File not found")
        print("  You can download it from: https://github.com/opencv/opencv/tree/master/data/haarcascades")
    
    # Check for sample dataset
    sample_csv = 'sample_fer2013.csv'
    if os.path.exists(sample_csv):
        print(f"✓ {sample_csv} - File exists")
    else:
        print(f"✗ {sample_csv} - File not found")
        print("  This file is needed for model training")
    
    # Summary
    print("\n----- Environment Check Summary -----")
    if all_modules_installed:
        print("✓ All required packages are installed")
    else:
        print("✗ Some packages are missing or need to be updated")
        print("  Run: pip install --user tensorflow keras pandas numpy pygame opencv-python imutils")
    
    print("\nFor project setup instructions, please refer to the README.md file.")

if __name__ == "__main__":
    check_environment() 