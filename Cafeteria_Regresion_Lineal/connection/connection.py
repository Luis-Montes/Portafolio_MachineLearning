from sqlalchemy import create_engine

server = 'localhost'
database = 'cafeteria'

engine = create_engine(
    f"mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)


query = """
select id_cliente, tipo_bebida, visitas_por_semana, 
gasto_promedio, es_estudiante,
(gasto_promedio * visitas_por_semana * 4) as gasto_total
from clientes;
"""