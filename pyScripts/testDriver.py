### For relatie import to work, the importing file must be executed as a module (python -m) or imported from a common ancestor package 
# Ex: import loader.test.test_read_partial -->  test_read_partial.py can import ..read_partial
import loader.test.test_read_partial