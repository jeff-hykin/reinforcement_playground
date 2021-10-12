bool1 = VariableTable.create(
        encoding=Encoding.boolean,
        is_top_level=True,
        boolean_value=True,
    )
bool2 = VariableTable.create(
        encoding=Encoding.boolean,
        is_top_level=True,
        boolean_value=True,
    )
string1 = VariableTable.create(
        encoding=Encoding.string,
        is_top_level=True,
        string_value="Hello World",
    )

map = VariableTable.create(
        encoding=Encoding.map,
        is_top_level=True,
    )

map_value1 = VariableTable.create(
        encoding=Encoding.number,
        number_value=99,
        is_top_level=False,
    )

element1 = MapElementTable.create(
        parent=map.id,
        key="x",
        variable=map_value1.id,
    )

map_value2 = VariableTable.create(
        encoding=Encoding.number,
        number_value=(2*3.1415926),
        is_top_level=False,
    )
    
element2 = MapElementTable.create(
        parent=map.id,
        key="y",
        variable=map_value2.id,
    )