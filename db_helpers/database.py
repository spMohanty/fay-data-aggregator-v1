"""Database utilities for Digital Epidemiology Lab."""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
from psycopg2.extensions import AsIs, register_adapter

DEBUG_LOGGING = False
CUR_DIR = Path(__file__).resolve().parent
LAB_DIR = CUR_DIR.parent

load_dotenv()
register_adapter(np.int64, AsIs)


class ParameterizedQuery:
    """A query with parameters and identifiers."""

    def __init__(
        self, t: str = "", p: Optional[List] = None, i: Optional[List] = None
    ) -> ParameterizedQuery:
        self.text = "" + t
        self.parameters = [] if p is None else p
        self.identifiers = [] if i is None else i

    def add_t(self, text: str):
        """Add text to the query."""
        self.text += f" {text} "

    def add_p(self, parameter: List):
        """Add a parameter to the query."""
        self.text += " %s "
        self.parameters.append(parameter)

    def add_i(self, identifier: List):
        """Add an identifier to the query."""
        self.text += " {} "
        self.identifiers.append(identifier)

    def add_q(self, other: ParameterizedQuery):
        """Add another ParameterizedQuery to this one."""
        self.text += other.text
        self.parameters += other.parameters
        self.identifiers += other.identifiers


class DB:
    """Database utility base class"""

    class Error(Exception):
        """Base class for errors raised from `DB`"""

    _schema_cache = {}

    def __init__(self, nickname: str) -> DB:
        if nickname == "FAY":
            host = os.getenv("FAY_DB_HOST")
            db = os.getenv("FAY_DB_NAME")
            user = os.getenv("FAY_DB_USER")
            password = os.getenv("FAY_DB_PW")
        elif nickname == "MFR":
            host = os.getenv("MFR_DB_HOST")
            db = os.getenv("MFR_DB_NAME")
            user = os.getenv("MFR_DB_USER")
            password = os.getenv("MFR_DB_PW")
        elif nickname == "FANAL":
            host = os.getenv("FANAL_DB_HOST")
            db = os.getenv("FANAL_DB_NAME")
            user = os.getenv("FANAL_DB_USER")
            password = os.getenv("FANAL_DB_PW")
        elif nickname == "MTDN":
            host = os.getenv("MTDN_DB_HOST")
            db = os.getenv("MTDN_DB_NAME")
            user = os.getenv("MTDN_DB_USER")
            password = os.getenv("MTDN_DB_PW")
        elif nickname == "FANAL_WRITE":
            host = os.getenv("FANAL_DB_HOST")
            db = os.getenv("FANAL_DB_NAME")
            user = os.getenv("FANAL_DB_MASTER_USER")
            password = os.getenv("FANAL_DB_MASTER_PW")
        else:
            raise ValueError("Invalid `db_name` value")

        self.connection_string = (
            f"host='{host}'"
            + f"dbname='{db}'"
            + f"user='{user}'"
            + f"password='{password}'"
        )

    @classmethod
    def get_schema(cls, nickname: str) -> dict:
        """Fetch the schema of a table.

        Arguments
        ---------
        nickname : str
            The nickname of the schema. It's the JSON filename, but not
            necessarily the same as the table name.

        Returns
        -------
        dict
            The schema, after being parsed from JSON.

        Notes
        -----
        Side effect: caches the schema in `DB._schemas`.
        """
        if nickname in cls._schema_cache:
            return cls._schema_cache[nickname]

        fpath = LAB_DIR / "db" / "schema" / f"{nickname}.json"
        with open(fpath, "r", encoding="utf-8") as file:
            res = json.load(file)
        cls._schema_cache[nickname] = res
        return res

    @classmethod
    def exec(
        cls,
        cur: psycopg2.extensions.cursor,
        q: str | ParameterizedQuery,
        p: Optional[List] = None,
        i: Optional[List] = None,
    ) -> pd.DataFrame:
        """Execute one query.

        May include multiple statements.

        Arguments
        ---------
        cur : psycopg2.extensions.cursor
            The database cursor.
        q : str | ParameterizedQuery
            The parameterized query. If `q` is a Parameterized query, it is
            expected that the parameters and identifiers are contained within,
            and therefore `p` and `i` should not be passed to this method too.
        p : list, optional
            The parameters to be substituted into the query string. (e.g. values
            such as `5` or `'hello'` that will be properly escaped)
        i : list, optional
            The identifiers to be substituted into the query string. (e.g.
            column names such as `user_id` that will be properly escaped)

        Returns
        -------
        pd.DataFrame
            The result of the query.
        """
        if isinstance(q, ParameterizedQuery):
            if p is not None or i is not None:
                raise ValueError(
                    "When passing a ParameterizedQuery, " + "`i` and `p` must be None"
                )
            query = q
            q = query.text
            p = query.parameters
            i = query.identifiers
        q = sql.SQL(q).format(*map(sql.Identifier, _list_wrap(i)))
        t0 = datetime.now() if DEBUG_LOGGING else None
        res = cur.execute(q, _list_wrap(p))
        if DEBUG_LOGGING:
            ms = int((datetime.now() - t0).total_seconds() * 1000)
            query_text = re.sub(r"\s+", " ", q.as_string(cur).strip())
            query_text = re.sub(r"\s([),])", "\\1", query_text)
            query_text = re.sub(r"\(\s", "(", query_text)
            color_red = "\033[35m"
            color_none = "\033[m"
            sys.stderr.write(f" {color_red}({ms}ms) {query_text}{color_none}\n")
        return res

    @classmethod
    def create_table(
        cls,
        cur: psycopg2.extensions.cursor,
        tbl_name: str,
        col_defs: List[List],
        sequence_cols: Optional[set] = None,
    ) -> None:
        """Create a table.

        Arguments
        ---------
        cur : psycopg2.extensions.cursor
            The database cursor.
        tbl_name : str
            The name of the table to create.
        col_defs : List[List]
            The column definitions. Each element is a list of 3 elements:
            * col_name : str
                The name of the column.
            * col_type : str
                The type of the column.
            * col_constraints : str
                The constraints of the column.
        sequence_cols : set, optional
            The names of the columns that should be sequences. Expects sequence
            named f"{tbl_name}_{col_name}_seq" to already exist. If `None`, no
            columns will be sequences.

        Returns
        -------
        Any
            The result of the execution.
        """
        if sequence_cols is None:
            sequence_cols = set()
        q = ParameterizedQuery()
        q.add_t("CREATE TABLE")
        q.add_i(tbl_name)
        q.add_t("(")
        for col_idx, [col_name, col_type, col_constraints] in enumerate(col_defs):
            if col_idx > 0:
                q.add_t(",")
            q.add_i(col_name)
            q.add_t(f"{col_type} {col_constraints}")  # is this safe?
            if col_name in sequence_cols:
                q.add_q(
                    ParameterizedQuery(
                        "DEFAULT nextval(%s)", p=[f"{tbl_name}_{col_name}_seq"]
                    )
                )
        q.add_t(")")
        return cls.exec(cur, q)

    @classmethod
    def drop_table(cls, cur: psycopg2.extensions.cursor, tbl_name: str):
        """Drop a table.

        This will also drop indexes associated with the table.

        Arguments
        ---------
        cur : psycopg2.extensions.cursor
            The database cursor.
        tbl_name : str
            The name of the table to drop.

        Returns
        -------
        Any
            The result of the execution.
        """
        return cls.exec(cur, "DROP TABLE {}", i=[tbl_name])

    @classmethod
    def create_sequence(
        cls, cur: psycopg2.extensions.cursor, tbl_name: str, col_name: str
    ) -> None:
        """Create a sequence.

        These are typically used for auto-incrementing primary keys.

        Arguments
        ---------
        cur : psycopg2.extensions.cursor
            The database cursor.
        tbl_name : str
            The name of the table to create the sequence for.
        col_name : str
            The name of the column the sequence is for.
        """
        cls.exec(cur, "CREATE SEQUENCE {}", i=[f"{tbl_name}_{col_name}_seq"])

    @classmethod
    def drop_sequence(
        cls, cur: psycopg2.extensions.cursor, tbl_name: str, col_name: str
    ) -> None:
        """Drop a sequence..

        Arguments
        ---------
        cur : psycopg2.extensions.cursor
            The database cursor.
        tbl_name : str
            The name of the table to drop the sequence for.
        """
        cls.exec(cur, "DROP SEQUENCE {}", i=f"{tbl_name}_{col_name}_seq")

    @classmethod
    def set_comment(
        cls, cur: psycopg2.extensions.cursor, tbl_name: str, col_name: str, comment: str
    ) -> None:
        """Set a comment on a column.

        Arguments
        ---------
        cur : psycopg2.extensions.cursor
            The database cursor.
        tbl_name : str
            The name of the table to set the comment on.
        col_name : str
            The name of the column to set the comment on.
        comment : str
            The comment to set.
        """
        q = "COMMENT ON COLUMN {}.{} IS %s"
        cls.exec(cur, q, p=[comment], i=[tbl_name, col_name])

    @classmethod
    def create_index(
        cls,
        cur: psycopg2.extensions.cursor,
        tbl_name: str,
        col_names: List[str],
        opts: dict,
    ):
        """Create an index.

        Arguments
        ---------
        cur : psycopg2.extensions.cursor
            The database cursor.
        tbl_name : str
            The name of the table to create the index on.
        col_names : List[str]
            The names of the columns to create the index on.
        opts : dict
            The options for the index. Currently only supports `unique`.
        """
        q = ParameterizedQuery()
        q.add_t("CREATE")
        if opts.get("unique", False):
            q.add_t("UNIQUE")
        q.add_t("INDEX")
        q.add_i(f"{tbl_name}_{'_'.join(col_names)}")
        q.add_t("ON")
        q.add_i(tbl_name)
        q.add_t("(")
        for col_idx, col_name in enumerate(col_names):
            if col_idx > 0:
                q.add_t(",")
            q.add_i(col_name)
        q.add_t(")")
        cls.exec(cur, q)

    @classmethod
    def insert_rows(
        cls,
        cur: psycopg2.extensions.cursor,
        tbl_name: str,
        df: pd.DataFrame,
        batch_size: int = 1000,
    ):
        """Insert batched rows into a table.

        Arguments
        ---------
        cur : psycopg2.extensions.cursor
            The database cursor.
        tbl_name : str
            The name of the table to insert rows into.
        df : pd.DataFrame
            The rows to insert.
        batch_size : int, optional
            The number of rows to insert at a time. (default 1000)

        Returns
        -------
        int
            The number of rows inserted.
        """
        q_one = "INSERT INTO {} ("
        q_one += ",".join(["{}"] * df.shape[1])
        q_one += ") VALUES %s"
        identifiers = [tbl_name] + df.columns.to_list()
        q_one = sql.SQL(q_one).format(*map(sql.Identifier, identifiers))
        q_one = q_one.as_string(cur)
        nrows = len(df)
        for _, batch in df.groupby(np.arange(nrows) // batch_size):
            q_batch = ";".join([q_one] * len(batch))
            p_batch = [tuple(row) for row in batch.values]
            cls.exec(cur, q_batch, p=p_batch)
        return nrows

    @classmethod
    def delete_rows(
        cls, cur: psycopg2.extensions.cursor, tbl_name: str, conds: List[List[Any]]
    ) -> int:
        """Delete rows from a table.

        Arguments
        ---------
        cur : psycopg2.extensions.cursor
            The database cursor.
        tbl_name : str
            The name of the table to delete rows from.
        conds : List[List[Any]]
            The conditions to delete rows by. Each element is a list of 3
            elements:
            * a : Any
                The left-hand side of the condition.
            * op : str
                The operator of the condition. Currently only supports `I EQ P`
                (identifier equal to parameter) and `I IN P` (identifier within
                parameter).
            * b : Any
                The right-hand side of the condition.

        Returns
        -------
        int
            The number of rows deleted.
        """
        if cur is None:
            raise cls.Error("Must provide a cursor")
        if len(tbl_name) == 0:
            raise cls.Error("Must provide a tbl_name")

        q = ParameterizedQuery()
        q.add_t("DELETE FROM")
        q.add_i(tbl_name)
        for idx, [a, op, b] in enumerate(conds):
            if idx == 0:
                q.add_t("WHERE")
            else:
                q.add_t("AND")
            if op == "I EQ P":  # identifier equal to parameter
                q.add_i(a)
                q.add_t("=")
                q.add_p(b)
            elif op == "I IN P":  # identifier within parameter
                q.add_i(a)
                q.add_t("IN")
                q.add_p(b)
            else:
                raise cls.Error(f"Unhandled operator `{op}`")
        cls.exec(cur, q)
        return cur.rowcount

    def table_exists(self, tbl_name: str) -> bool:
        """Check if a table exists.

        Arguments
        ---------
        tbl_name : str
            The name of the table to check.

        Returns
        -------
        bool
            Whether the table exists.
        """
        fpath = LAB_DIR / "db" / "sql" / "db_table_exists.sql"
        with open(fpath, "r", encoding="utf-8") as file:
            return self.select_val(file.read(), p=[tbl_name])

    def setup_table(self, schema: dict) -> bool:
        """Completely set up a new table.

        If it already exists, do nothing. Otherwise, create the table, its
        sequences, indexes, comments, etc.

        Arguments
        ---------
        schema : dict
            The schema of the table to set up. See examples in `db/schema`.

        Returns
        -------
        bool
            Whether the table was created. (`False` if it existed already)
        """
        tbl_name = schema["name"]
        if self.table_exists(tbl_name):
            return False

        cls = type(self)

        def _tx(cur):
            # create sequences
            sequences = set(schema.get("sequences", []))
            for col_name in sequences:
                cls.create_sequence(cur, tbl_name, col_name)
            # create table
            col_defs = schema["columns"]
            cls.create_table(cur, tbl_name, col_defs, sequences)
            # set comments
            comments = schema.get("comments", {})
            for col_name, comment in comments.items():
                cls.set_comment(cur, tbl_name, col_name, comment)
            # create indexes
            for raw_index in schema.get("indexes", []):
                if isinstance(raw_index, dict):
                    opts = raw_index
                    col_names = _list_wrap(opts.pop("col"))
                else:
                    opts = {}
                    col_names = _list_wrap(raw_index)
                cls.create_index(cur, tbl_name, col_names, opts)

        self.execute_tx(_tx)
        return True

    def teardown_table(self, schema: dict) -> bool:
        """Completely tear down a table.

        If it does not exist, do nothing. Otherwise, drop the table, its
        sequences, indexes, etc.

        Arguments
        ---------
        schema : dict
            The schema of the table to tear down. See examples in `db/schema`.

        Returns
        -------
        bool
            Whether the table was dropped. (`False` if it didn't exist)
        """
        tbl_name = schema["name"]
        if not self.table_exists(tbl_name):
            return False

        cls = type(self)

        def _tx(cur):
            cls.drop_table(cur, tbl_name)
            for col_name in schema.get("sequences", []):
                cls.drop_sequence(cur, tbl_name, col_name)

        self.execute_tx(_tx)
        return True

    def execute_tx(self, func: Callable[[psycopg2.extensions.cursor], Any]) -> Any:
        """Execute database commands in a transaction.

        Connects to the db, starts a tx, calls the given function, and commits
        the tx. If an exception is raised in `func`, rolls back the tx, closes
        the connection, and re-raises the exception.

        Arguments
        ---------
        func : Callable[[psycopg2.extensions.cursor], Any]
            The function to call within the tx. Takes 1 argument `cur`, the db
            cursor.

        Returns
        -------
        any
            The return value of `func`.
        """
        con = psycopg2.connect(self.connection_string)
        cur = con.cursor()
        try:
            res = func(cur)
            con.commit()
        except Exception as e:
            con.rollback()
            cur.close()
            con.close()
            raise e
        return res

    def select(
        self,
        q: str | ParameterizedQuery,
        p: Optional[List] = None,
        i: Optional[List] = None,
    ) -> pd.DataFrame:
        """Execute a SELECT query.

        Arguments
        ---------
        See `DB.exec`

        Returns
        -------
        pd.DataFrame
            The result of the query.
        """

        def _tx(cur):
            type(self).exec(cur, q, p, i)
            col_names = [desc[0] for desc in cur.description]
            col_types = [desc[1] for desc in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=col_names)
            _fix_df_dtypes(df, col_types)
            return df

        return self.execute_tx(_tx)

    def select_col(
        self,
        q: str | ParameterizedQuery,
        p: Optional[List] = None,
        i: Optional[List] = None,
    ) -> pd.Series:
        """Execute a SELECT query.

        Arguments
        ---------
        See `DB.exec`

        Returns
        -------
        pd.Series
            Column 0 of result of the query.
        """
        return self.select(q, p, i).iloc[:, 0]

    def select_val(
        self,
        q: str | ParameterizedQuery,
        p: Optional[List] = None,
        i: Optional[List] = None,
    ) -> Any:
        """Execute a SELECT query.

        Arguments
        ---------
        See `DB.exec`

        Returns
        -------
        Any
            The first value of result of the query.
        """
        return self.select(q, p, i).iloc[0, 0]

    def delete(
        self,
        q: str | ParameterizedQuery,
        p: Optional[List] = None,
        i: Optional[List] = None,
    ) -> int:
        """Execute a DELETE query.

        Arguments
        ---------
        See `DB.exec`

        Returns
        -------
        int
            The number of rows deleted.
        """

        def _tx(cur):
            type(self).exec(cur, q, p, i)
            return cur.rowcount

        return self.execute_tx(_tx)


class MastodonDb(DB):
    """Subclass of `DB` for the Mastodon Streamer database"""

    def __init__(self):
        super().__init__("MTDN")


class FayDb(DB):
    """Subclass of `DB` for the F&Y database"""

    def __init__(self):
        super().__init__("FAY")


class MfrDb(DB):
    """Subclass of `DB` for the MFR database"""

    def __init__(self):
        super().__init__("MFR")


class AnalyticsDb(DB):
    """Subclass of `DB` for the FANAL database"""

    def __init__(self, mode="r"):
        if mode == "r":
            super().__init__("FANAL")
        elif mode == "w":
            super().__init__("FANAL_WRITE")
        else:
            raise DB.Error(f"Invalid mode `{mode}`")


def _list_wrap(x: Any) -> List[Any]:
    """Wrap the argument (or its value(s)) into a list."""
    if x is None:
        res = []
    elif isinstance(x, list):
        res = x
    elif isinstance(x, tuple):
        res = list(x)
    elif isinstance(x, pd.DataFrame):
        res = list(x.iloc[:, 0].values)
    elif isinstance(x, pd.Series):
        res = list(x.values)
    elif isinstance(x, dict):
        res = list(x.values())
    else:
        res = [x]
    return res


def _fix_df_dtypes(df, col_types: List[int]):
    """Fix the datatypes of a DataFrame that was returned from a DB query."""
    if len(df) == 0:
        return

    for idx, type_code in enumerate(col_types):
        col = df.columns[idx]
        if type_code == psycopg2.extensions.DATE:
            # imported as 'object', convert to date
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")
        if type_code == 1114:  # PSQL TIMESTAMP WITHOUT TIME ZONE
            # imported as tz-unaware datetime, convert to UTC
            df[col] = df[col].dt.tz_localize("UTC")
