def Db(name="records"):
from peewee import SqliteDatabase, Model, AutoField, BigAutoField, BooleanField, SmallIntegerField, IntegerField, DoubleField, CharField, TextField, ForeignKeyField, DeferredForeignKey, DateTimeField, Case, JOIN, fn



db = SqliteDatabase(name+'.dont-sync.db')
class BaseModel(Model):
    class Meta:
        database = db

class Interpretation(BaseModel):
    name = CharField(primary_key=True)

# create an encoding
import dataclasses
@dataclasses.dataclass
class Encoding:
    null = 0
    boolean = 1
    number = 2
    string = 3
    list = 4
    map = 5
    variable = 6

class VariableTable(BaseModel):
    id       = BigAutoField(primary_key=True)
    is_top_level = BooleanField(null=True)
    encoding = SmallIntegerField(default=0)
    
    # primitive values
    boolean_value  = BooleanField(null=True)
    number_value   = DoubleField(null=True)
    string_value   = TextField(null=True)
    # list_elements
    # map_elements
    
    # linked value
    variable_value = DeferredForeignKey("VariableTable", null=True)
    # allow different interpretations (such as interpreting a string as regex, or as a datetime)
    interpretation = ForeignKeyField(Interpretation, null=True)
    
    @property
    def type(self):
        if self.encoding == Encoding.null:
            return type(None)
        elif self.encoding == Encoding.boolean:
            return bool
        elif self.encoding == Encoding.number:
            return float
        elif self.encoding == Encoding.string:
            return str
        elif self.encoding == Encoding.list:
            return list
        elif self.encoding == Encoding.map:
            return dict
        elif self.encoding == Encoding.variable:
            return VariableTable
    
    @property
    def value(self):
        if self.encoding == Encoding.null:
            return None
        elif self.encoding == Encoding.boolean:
            return self.boolean_value
        elif self.encoding == Encoding.number:
            return self.number_value
        elif self.encoding == Encoding.string:
            return self.string_value
        elif self.encoding == Encoding.list:
            return self.list_elements
        elif self.encoding == Encoding.map:
            return self.map_elements
        elif self.encoding == Encoding.variable:
            return self.variable_value

class ListElementTable(BaseModel):
    id       = BigAutoField(primary_key=True)
    parent   = ForeignKeyField(VariableTable, backref="list_elements")
    key      = IntegerField(default=0)
    variable = ForeignKeyField(VariableTable)

class MapElementTable(BaseModel):
    id       = BigAutoField(primary_key=True)
    parent   = ForeignKeyField(VariableTable, backref="map_elements")
    key      = TextField()
    variable = ForeignKeyField(VariableTable)

import numbers
class DbValue():
    @classmethod
    def new(DbValue, value):
        # almost all logic is inside of dangerous_create_value()
        # that is because it calls itself recursively copies everything in the other value (then points to it) 
        # (so it needs 99% of logic)
        # the only thing not inside it, is the with atomic:
        # (which shouldn't be recursive, and is also what makes it safe again)
        def dangerous_create_value(value, is_top_level=False):
            db_entry = None
            # null
            if type(value) == type(None):
                db_entry = VariableTable.create(
                    encoding=Encoding.null,
                    is_top_level=is_top_level,
                )
            # bool
            elif isinstance(value, bool):
                db_entry = VariableTable.create(
                    encoding=Encoding.boolean,
                    boolean_value=value,
                    is_top_level=is_top_level,
                )
            # number
            elif isinstance(value, numbers.Number):
                db_entry = VariableTable.create(
                    encoding=Encoding.number,
                    number_value=value,
                    is_top_level=is_top_level,
                )
            # string
            elif isinstance(value, str):
                db_entry = VariableTable.create(
                    encoding=Encoding.string,
                    string_value=value,
                    is_top_level=is_top_level,
                )
            # list
            elif isinstance(value, (list, tuple)):
                # create the parent variable
                db_entry = VariableTable.create(
                    encoding=Encoding.list,
                    is_top_level=is_top_level,
                )
                # ensure everything is a DbValue
                all_elements_as_db_values = [
                    # if its not already a DbValue, add it as a DbValue
                    (each if isinstance(each, DbValue) else dangerous_create_value(each))
                        for each in value
                ]
                # add all the elements of the list to the list table
                for each_index, each_value in enumerate(all_elements_as_db_values):
                    ListElementTable.create(
                        parent=db_entry.id,
                        key=each_index,
                        variable=each_value.variable_id,
                    )
            elif isinstance(value, dict):
                # TODO: fixme
                pass
            elif isinstance(value, DbValue):
                # TODO: fixme
                pass
                
            return Value(variable=db_entry)
            
        # (e.g. if something breaks, undo the partial changes)
        with db.atomic():
            return dangerous_create_value(value, is_top_level=True)
    
    # 
    # properties
    # 
    variable = None
    
    # 
    # methods
    # 
    def __init__(self, variable):
        self.variable = variable
    
    # this value points to another value
    def link(self, another_db_value):
        pass
    
    # this value points to a copy of another value
    def link_shallow_copy(self, another_db_value):
        pass
    
    # this value recursively gathers all values in the other value (into a set)
    # then it creates a copy of all of them, replacing their forign keys with the newly created forign keys
    # (note: this could easily result it duplicating the whole database if there is a child with a reference to root)
    def link_deep_copy(self, another_db_value):
        pass
    
    @property
    def is_primitive(self):
        return self.encoding <= Encoding.string
    
    @property
    def value(self,):
        if self.variable.encoding == Encoding.null:
            return None
        elif self.variable.encoding == Encoding.boolean:
            return self.boolean_value
        elif self.variable.encoding == Encoding.number:
            return self.number_value
        elif self.variable.encoding == Encoding.string:
            return self.string_value
        elif self.variable.encoding == Encoding.list:
            ordered_list_elements = self.list_elements.order_by(
                    ListElementTable.key
                ).select(
                    ListElementTable.variable
                ).join(
                    # which data to combine into existing table
                    VariableTable,
                    # what attribute (on the element) should be retrieved
                    attr='variable'
                    # which key is being looked-up
                    on=(VariableTable.id == ListElementTable.variable),
                )
            # iterator of variables
            return (DbValue(variable=each.variable) for each in ordered_list_elements)
        elif self.variable.encoding == Encoding.map:
            # FIXME: todo
            pass
        elif self.variable.encoding == Encoding.variable:
            # get the value it points to
            return DbValue(variable=self.variable.variable_value)
    
    @property
    def recursively_pure_value(self):
        if self.is_primitive:
            return self.value
        else:
            if self.variable.encoding == Encoding.list:
                # FIXME: a list can contain itself, so catch that instead of causing a stack overflow
                return (each_db_value.recursively_pure_value for each_db_value in self.list_elements)
            elif self.variable.encoding == Encoding.map:
                # FIXME: todo
                pass
            elif self.variable.encoding == Encoding.variable:
                return self.value.recursively_pure_value
    
    def values(self):
        if self.variable.encoding == Encoding.list:
            ordered_list_elements = self.list_elements.order_by(
                    ListElementTable.key
                ).select(
                    ListElementTable.variable
                ).join(
                    # which data to combine into existing table
                    VariableTable,
                    # what attribute (on the element) should be retrieved
                    attr='variable',
                    # which key is being looked-up
                    on=(VariableTable.id == ListElementTable.variable),
                )
            
            return (each.variable.value for each in ordered_list_elements)

# TODO: select to value
# def selection_to_value(selection):
    


db.connect()
db.create_tables([
    Interpretation, VariableTable, ListElementTable, MapElementTable
])

# select all possible parents
top_level_vars = VariableTable.select().where(VariableTable.is_top_level==True)

aliaser = Case(None, (
        (VariableTable.encoding == Encoding.boolean, VariableTable.boolean_value == 1),
        (VariableTable.encoding == Encoding.number, VariableTable.number_value),
        (VariableTable.encoding == Encoding.string, VariableTable.string_value),
        (VariableTable.encoding == Encoding.map, [1,2,3]),
    ), None)

boolean_vars = top_level_vars.where(
        VariableTable.encoding == Encoding.boolean,
    ).select(VariableTable, aliaser.alias("v"))

number_vars = VariableTable.select().where(
        VariableTable.encoding == Encoding.number,
    ).select(VariableTable, aliaser.alias("v")).prefetch()

string_vars = VariableTable.select().where(
        VariableTable.encoding == Encoding.string,
    ).select(VariableTable, aliaser.alias("v")).prefetch()

map_vars = VariableTable.select().where(
        VariableTable.encoding == Encoding.map,
    ).select(VariableTable, aliaser.alias("v")).join(
        MapElementTable,
        on=(VariableTable.id == MapElementTable.parent),
    ).prefetch(MapElementTable)
    

all_vars = number_vars | boolean_vars | string_vars | map_vars
for each in all_vars: print(vars(each))
for each in boolean_vars: print(vars(each))
for each in number_vars: print(vars(each))
for each in string_vars: print(vars(each))

map_vars = top_level_vars.where(
        VariableTable.encoding == Encoding.map,
    ).join(
        MapElementTable,
        JOIN.INNER,
        attr="element",
        on=(VariableTable.id == MapElementTable.parent),
    ).group_by(
        VariableTable.id, 
    ).select().prefetch(MapElementTable)

vars_with_key = lambda key: VariableTable.select().where(VariableTable.id.in_(
       MapElementTable.select().where(MapElementTable.key == key).where(MapElementTable.parent.in_(top_level_vars)).join(
            # which data to combine into existing table
            VariableTable,
            # what attribute (on the element) should be retrieved
            attr='parent',
            # which key is being looked-up
            on=(VariableTable.id == MapElementTable.parent),
        ).group_by(MapElementTable.parent).select(MapElementTable.parent) 
    )).join(MapElementTable, on=(VariableTable.id == MapElementTable.parent))

# TODO: get the value of each var and key specified
# for each key in keys, query.join(MapElementTable, attr="_"+key)
# recursive self join https://stackoverflow.com/questions/1757260/simplest-way-to-do-a-recursive-self-join

# all the map elements with that key => find the parents and join, but select the parents
query = list_elements_with_value = ListElementTable.select(
    # get the list elements for the top level
    ).where(
        ListElementTable.parent.in_(top_level)
    ).join(
        # which data to combine into existing table
        VariableTable,
        # what attribute (on the element) should be retrieved
        attr='variable',
        # which key is being looked-up
        on=(VariableTable.id == ListElementTable.variable),
    )


# all variables that are maps and have "x" as keys
query = (
        each.parent for each in MapElementTable.select().where(MapElementTable.key == "x").where(MapElementTable.parent.in_(top_level_vars)).join(
            # which data to combine into existing table
            VariableTable,
            # what attribute (on the element) should be retrieved
            attr='parent',
            # which key is being looked-up
            on=(VariableTable.id == MapElementTable.parent),
        ).group_by(MapElementTable.parent).select(MapElementTable.parent)
    )

top_level_map_elements = MapElementTable.select().where(MapElementTable.key == "x").where(MapElementTable.parent.in_(top_level_vars)).join(
            # which data to combine into existing table
            VariableTable,
            # what attribute (on the element) should be retrieved
            attr='parent',
            # which key is being looked-up
            on=(VariableTable.id == MapElementTable.parent),
        ).group_by(MapElementTable.parent).select(MapElementTable.parent)
        
query = VariableTable.select().where(VariableTable.id.in_(
       MapElementTable.select().where(MapElementTable.key == "x").where(MapElementTable.parent.in_(top_level_vars)).join(
            # which data to combine into existing table
            VariableTable,
            # what attribute (on the element) should be retrieved
            attr='parent',
            # which key is being looked-up
            on=(VariableTable.id == MapElementTable.parent),
        ).group_by(MapElementTable.parent).select(MapElementTable.parent) 
    )) 

key = "x"
         
query = MapElementTable.select().where(MapElementTable.key == key).where(MapElementTable.parent.in_(top_level_vars)).join(
            # which data to combine into existing table
            VariableTable,
            # what attribute (on the element) should be retrieved
            attr='parent',
            # which key is being looked-up
            on=(VariableTable.id == MapElementTable.parent),
        ).group_by(VariableTable).switch(VariableTable).select()
for each in where_key_is("x"): print(hasattr(each, "value") and each.type)
for each in query: print(hasattr(each, "value") and each.type)

VariableTable.select(
    # get all the list elements
    ).join(
        list_elements_with_value, attr='variable', on=(VariableTable.id == list_elements_with_value.variable)
    )




return locals()
