-- Generated from Settings; regenerate with python -m src.cli databricks-bootstrap.
-- Run as a Unity Catalog administrator or principal with CREATE CATALOG.
-- Schema creation requires USE CATALOG and CREATE SCHEMA on the catalog.
-- Volume creation requires USE SCHEMA and CREATE VOLUME on the raw schema.
-- Runtime principal needs USE CATALOG, USE SCHEMA, READ/WRITE VOLUME,
-- CREATE TABLE/MODIFY/SELECT on processing schemas and access to Serverless Jobs.
CREATE CATALOG IF NOT EXISTS `shopee`;
CREATE SCHEMA IF NOT EXISTS `shopee`.`raw`;
CREATE SCHEMA IF NOT EXISTS `shopee`.`bronze`;
CREATE SCHEMA IF NOT EXISTS `shopee`.`silver`;
CREATE SCHEMA IF NOT EXISTS `shopee`.`gold`;
CREATE VOLUME IF NOT EXISTS `shopee`.`raw`.`landing`;
