# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Answer(models.Model):
    a_id = models.AutoField(primary_key=True)
    a_answer = models.CharField(max_length=255)
    q = models.ForeignKey('Question', models.DO_NOTHING)
    i_id = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'answer'


class Ciconnect(models.Model):
    ci_id = models.AutoField(primary_key=True)
    cr = models.ForeignKey('Crawlrecord', models.DO_NOTHING)
    i = models.ForeignKey('Instrument', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'ciconnect'


class Crawlrecord(models.Model):
    cr_id = models.AutoField(primary_key=True)
    cr_keyword = models.CharField(max_length=255)
    cr_num = models.IntegerField()
    cr_time = models.DateTimeField()
    u = models.ForeignKey('User', models.DO_NOTHING)
    cr_status = models.IntegerField()
    cr_check = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'crawlrecord'


class File(models.Model):
    f_id = models.AutoField(primary_key=True)
    u = models.ForeignKey('User', models.DO_NOTHING)
    f_name = models.CharField(max_length=255)
    f_path = models.CharField(max_length=255)
    f_time = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'file'


class Filerecord(models.Model):
    fr_id = models.AutoField(primary_key=True)
    f_id = models.CharField(max_length=255)
    fr_time = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'filerecord'


class Instrument(models.Model):
    i_id = models.AutoField(primary_key=True)
    i_title = models.CharField(max_length=255)
    i_path = models.CharField(max_length=255)
    cr_keyword = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = 'instrument'


class Question(models.Model):
    q_id = models.AutoField(primary_key=True)
    q_name = models.CharField(max_length=255)
    r_id = models.IntegerField()
    r_flag = models.IntegerField()
    q_status = models.IntegerField()
    q_check = models.IntegerField()
    u = models.ForeignKey('User', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'question'


class User(models.Model):
    u_id = models.AutoField(primary_key=True)
    u_name = models.CharField(max_length=255)
    u_pwd = models.CharField(max_length=255)
    u_email = models.CharField(max_length=255)
    u_phone = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = 'user'
