"""Import first in a stdio entry point: keeps stdout for the protocol (see ``rpc.reserve_stdout``)."""

from app.services.harness.rpc import reserve_stdout

reserve_stdout()
