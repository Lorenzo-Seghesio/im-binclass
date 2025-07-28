# im-binclass
## Faulty parts binary recognition using machine parameters in Injection Moulding

Used Python 3.12.3
Needed packages are specified in requirements.txt

### How to run the code
You can run the code with the following command
'''
python IM_Binary_Quality_Recognition.py --arg1
'''
The argument `arg1` can be:
- '--first_data' - to use the first dataset only with machine parameters
- '--first_data_full' - to use the first dataset with machine parameters and sensor measurements
- '--pp_data' - to use new dataset with only PP material data
- '--abs_data' - to use new dataset with only ABS material data
- If I leave 'arg1' empty then by default the code uses the new dataset with both PP and ABS data 