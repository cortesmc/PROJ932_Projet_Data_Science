# Generated by Django 5.1.3 on 2024-11-14 20:02

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Node',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('node_type', models.CharField(choices=[('person', 'Personne'), ('organization', 'Organisation'), ('location', 'Lieu')], max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Edge',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('relation_type', models.CharField(max_length=50)),
                ('from_node', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='from_edges', to='graphs.node')),
                ('to_node', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='to_edges', to='graphs.node')),
            ],
        ),
    ]