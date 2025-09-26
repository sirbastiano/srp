# Define base paths relative to project root
PROJECT_ROOT := /Data_large/marine/PythonProjects/SAR/sarpyx
OUTPUT_DIR := $(PROJECT_ROOT)/data/xview3/output

# Define target directories
SLCFILE_DIR := $(OUTPUT_DIR)/slcfile
GRDFILE_DIR := $(OUTPUT_DIR)/grdfile
XMLFILE_DIR := $(OUTPUT_DIR)/xmlfile

.PHONY: clean clean-slc clean-grd clean-xml help

# Default target
help:
    @echo "Available targets:"
    @echo "  clean     - Remove all output files"
    @echo "  clean-slc - Remove SLC files only"
    @echo "  clean-grd - Remove GRD files only" 
    @echo "  clean-xml - Remove XML files only"
    @echo "  help      - Show this help message"

# Clean all output directories
clean: clean-slc clean-grd clean-xml
    @echo "Cleaned all output directories"

# Clean individual directories with safety checks
clean-slc:
    @if [ -d "$(SLCFILE_DIR)" ]; then \
        echo "Cleaning $(SLCFILE_DIR)..."; \
        rm -rf "$(SLCFILE_DIR)"/*; \
    else \
        echo "Directory $(SLCFILE_DIR) does not exist, skipping..."; \
    fi

clean-grd:
    @if [ -d "$(GRDFILE_DIR)" ]; then \
        echo "Cleaning $(GRDFILE_DIR)..."; \
        rm -rf "$(GRDFILE_DIR)"/*; \
    else \
        echo "Directory $(GRDFILE_DIR) does not exist, skipping..."; \
    fi

clean-xml:
    @if [ -d "$(XMLFILE_DIR)" ]; then \
        echo "Cleaning $(XMLFILE_DIR)..."; \
        rm -rf "$(XMLFILE_DIR)"/*; \
    else \
        echo "Directory $(XMLFILE_DIR) does not exist, skipping..."; \
    fi