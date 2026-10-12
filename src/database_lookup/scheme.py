from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document


class Scheme:
    def __init__(self, database_url: str, admin_id: str):
        self.engine = create_engine(database_url)
        self.inspector = inspect(self.engine)
        self.db = SQLDatabase(self.engine, sample_rows_in_table_info=0)
        self.admin_id = admin_id

    def create_scheme(self) -> list[Document]:
        scheme_docs = []

        try:
            tables = self.inspector.get_table_names()

            for table in tables:
                clean_table = table.replace("_", " ")

                columns = self.inspector.get_columns(table)

                clean_columns = []

                for col in columns:
                    clean_col = col['name'].replace("_", " ")
                    clean_columns.append(clean_col)

                doc = f"Table: {clean_table}.\nColumns: {', '.join(clean_columns)}"

                scheme_docs.append(
                    Document(
                        page_content=doc,
                        metadata={
                            "table_id": f"{self.admin_id}.{table}",
                            "admin_id": self.admin_id,
                            "table_name": table,
                            "dialect": self.db.dialect,
                            # exact CREATE TABLE text the LLM will use to write SQL
                            "table_info": self.db.get_table_info([table]),
                        }
                    )
                )
        finally:
            self.engine.dispose()

        return scheme_docs