add_mlir_dialect_library(MLIRNkipy
        NkipyDialect.cpp
        NkipyOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/nkipy

        DEPENDS
        MLIRNkipyOpsIncGen
        MLIRNkipyAttrsIncGen
        MLIRNkipyEnumsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	MLIRDestinationStyleOpInterface
	MLIRLinalgDialect
	MLIRMemRefDialect
	MLIRTensorDialect
	MLIRTilingInterface
	)
