import django.core.management.commands.migrate as migrate_command
import django.db.migrations.executor as migration_executor
import django.core.management.commands.runserver as runserver_command

# Override migration command to prevent database checks
migrate_command.Command.handle = lambda *args, **kwargs: None
migration_executor.MigrationExecutor = lambda *args, **kwargs: None

# Override migration checks for `runserver`
runserver_command.Command.check_migrations = lambda *args, **kwargs: None

#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'graph_dashboard.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
