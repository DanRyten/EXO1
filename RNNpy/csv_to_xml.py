import xml.etree.ElementTree as ET
from xml.dom import minidom

def convert_to_xml(input_file, output_file):
    # Create the root element
    root = ET.Element("root")
    
    # Read the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Process the lines in pairs (title and data)
    for i in range(0, len(lines), 2):
        title = lines[i].strip()
        if i+1 < len(lines):
            data = lines[i+1].strip()
        else:
            data = ""
        
        # Create an element for each title-data pair
        item = ET.Element("item")
        title_element = ET.SubElement(item, "title")
        title_element.text = title
        data_element = ET.SubElement(item, "data")
        data_element.text = data
        
        # Append the item to the root element
        root.append(item)
    
    # Create a tree from the root element and write it to the output file
    tree = ET.ElementTree(root)
    #tree.write(output_file, encoding='utf-8', xml_declaration=True)

    xml_str = ET.tostring(root, encoding='utf-8')

    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="    ")

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(pretty_xml)
