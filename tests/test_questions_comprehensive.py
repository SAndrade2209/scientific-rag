"""
Test exhaustivo para el sistema RAG - Preguntas sobre pavimentos de aeropuertos
Categorías:
1. Preguntas específicas de un solo documento (detalladas)
2. Preguntas que requieren múltiples documentos
3. Preguntas comparativas
4. Preguntas técnicas con términos específicos
5. Preguntas que NO deben tener respuesta (fuera de alcance)
"""

test_questions = [
    # ============================================================================
    # CATEGORÍA 1: PREGUNTAS MUY ESPECÍFICAS DE UN SOLO DOCUMENTO
    # ============================================================================
    {
        "id": "Q1_SPECIFIC",
        "category": "single_document_detailed",
        "expected_doc": "006. Field Evaluation of Ultra-High Pressure Water Systems",
        "question": "What specific ultra-high pressure water (UHPW) equipment systems were tested at Edwards Air Force Base for runway rubber removal, and what were their key technical specifications including operating pressures?",
        "difficulty": "hard",
        "requires_details": True,
        "answer_should_include": ["Cyclone 4006", "Stripe Hog SH8000R", "Trackjet", "pressure specifications", "equipment names"]
    },

    {
        "id": "Q2_SPECIFIC",
        "category": "single_document_detailed",
        "expected_doc": "015. Automatically melting snow on airport cement concrete pavement",
        "question": "In the carbon fiber grille snow melting experiment, what was the exact power input used (in W/m²), how thick was the snow layer, and how long did it take to melt completely when the air temperature ranged from -3°C to -1°C?",
        "difficulty": "hard",
        "requires_details": True,
        "answer_should_include": ["350 W/m²", "2.7 cm", "2 hours", "temperature increment"]
    },

    {
        "id": "Q3_SPECIFIC",
        "category": "single_document_detailed",
        "expected_doc": "012. Finite element framework for runway friction",
        "question": "What specific aircraft tire model was used in the finite element simulation study, including its tire designation, tread width, nominal diameter, and rated inflation pressure?",
        "difficulty": "medium",
        "requires_details": True,
        "answer_should_include": ["32 × 8.8 Type VII", "223.5 mm", "812.8 mm", "2,200 kPa"]
    },

    {
        "id": "Q4_SPECIFIC",
        "category": "single_document_detailed",
        "expected_doc": "017. Time Domain Reflectometry LTPP",
        "question": "Describe the complete workflow used in the LTPP Seasonal Monitoring Program to convert raw TDR waveform data into gravimetric moisture content values, including what processing software was used.",
        "difficulty": "hard",
        "requires_details": True,
        "answer_should_include": ["TDR waveform", "apparent probe length", "dielectric constant", "volumetric moisture", "gravimetric moisture", "Moister"]
    },

    {
        "id": "Q5_SPECIFIC",
        "category": "single_document_detailed",
        "expected_doc": "006. Field Evaluation of Ultra-High Pressure Water Systems",
        "question": "What friction measurement devices were used to evaluate pavement skid resistance before and after rubber removal operations, and what standards do they follow?",
        "difficulty": "medium",
        "requires_details": True,
        "answer_should_include": ["Circular track meter", "Dynamic friction tester", "GripTester", "Hydrotimer", "NASA grease smear", "ASTM"]
    },

    # ============================================================================
    # CATEGORÍA 2: PREGUNTAS QUE REQUIEREN MÚLTIPLES DOCUMENTOS
    # ============================================================================
    {
        "id": "Q6_MULTI",
        "category": "multi_document_integration",
        "expected_docs": ["012. Finite element", "006. Rubber removal"],
        "question": "How does runway surface texture affect aircraft braking friction during wet conditions, and why is rubber removal important for maintaining adequate friction levels?",
        "difficulty": "medium",
        "requires_integration": True,
        "answer_should_include": ["macrotexture", "microtexture", "rubber deposits", "water drainage", "hydroplaning", "friction reduction"]
    },

    {
        "id": "Q7_MULTI",
        "category": "multi_document_integration",
        "expected_docs": ["015. Carbon fiber grille", "006. Conductive asphalt"],
        "question": "Compare the different electrically heated pavement technologies for snow melting on airport surfaces, including their power requirements, construction methods, and effectiveness.",
        "difficulty": "hard",
        "requires_integration": True,
        "answer_should_include": ["carbon fiber grille", "conductive asphalt", "power density", "W/m²", "heating cables", "construction"]
    },

    {
        "id": "Q8_MULTI",
        "category": "multi_document_integration",
        "expected_docs": ["001. Winter runway friction NASA", "005. Impact of snowfall", "015. Carbon fiber"],
        "question": "What are the operational impacts of winter weather on airport runway operations, and what technological solutions have been developed to mitigate delays and safety issues?",
        "difficulty": "hard",
        "requires_integration": True,
        "answer_should_include": ["flight delays", "friction coefficient", "snow removal", "heated pavement", "safety", "operations"]
    },

    {
        "id": "Q9_MULTI",
        "category": "multi_document_integration",
        "expected_docs": ["012. Finite element", "006. Rubber removal", "001. Winter friction"],
        "question": "Explain the relationship between pavement surface texture (microtexture and macrotexture), water film depth, and the development of hydroplaning conditions for aircraft tires.",
        "difficulty": "hard",
        "requires_integration": True,
        "answer_should_include": ["microtexture", "macrotexture", "adhesional friction", "hysteresis", "water depth", "hydroplaning", "drainage"]
    },

    {
        "id": "Q10_MULTI",
        "category": "multi_document_integration",
        "expected_docs": ["017. TDR LTPP", "019. SHRP Pavements Toledo"],
        "question": "How were Time Domain Reflectometry (TDR) sensors deployed and calibrated in the SHRP pavement seasonal monitoring studies, and what were the typical ranges of moisture variation observed?",
        "difficulty": "hard",
        "requires_integration": True,
        "answer_should_include": ["TDR", "sensor installation", "calibration", "moisture content", "seasonal variation", "10-15%"]
    },

    # ============================================================================
    # CATEGORÍA 3: PREGUNTAS COMPARATIVAS Y DE ANÁLISIS
    # ============================================================================
    {
        "id": "Q11_COMPARE",
        "category": "comparative_analysis",
        "expected_docs": ["Multiple"],
        "question": "Compare the advantages and disadvantages of chemical deicing methods versus heated pavement systems for airport snow management, considering cost, environmental impact, and operational effectiveness.",
        "difficulty": "medium",
        "answer_should_include": ["chemicals", "heated pavement", "cost", "environment", "corrosion", "energy", "maintenance"]
    },

    {
        "id": "Q12_COMPARE",
        "category": "comparative_analysis",
        "expected_docs": ["012. Finite element"],
        "question": "What are the key differences in friction behavior between aircraft tires and PIARC ground measurement tires, especially under extreme conditions of high inflation pressure and deep water?",
        "difficulty": "hard",
        "answer_should_include": ["aircraft tire", "PIARC", "inflation pressure", "water depth", "friction coefficient", "differences"]
    },

    {
        "id": "Q13_COMPARE",
        "category": "comparative_analysis",
        "expected_docs": ["006. Rubber removal"],
        "question": "Compare the effectiveness of the three different UHPW rubber removal systems tested (Cyclone 4006, Stripe Hog SH8000R, and Trackjet) in terms of friction improvement and operational characteristics.",
        "difficulty": "hard",
        "answer_should_include": ["Cyclone", "Stripe Hog", "Trackjet", "friction", "comparison", "effectiveness"]
    },

    # ============================================================================
    # CATEGORÍA 4: PREGUNTAS TÉCNICAS CON TERMINOLOGÍA ESPECÍFICA
    # ============================================================================
    {
        "id": "Q14_TECHNICAL",
        "category": "technical_terminology",
        "expected_docs": ["012. Finite element"],
        "question": "What is the Coupled Eulerian Lagrangian (CEL) technique, and why is it advantageous for modeling fluid-tire interactions in wet runway friction simulations?",
        "difficulty": "hard",
        "answer_should_include": ["CEL", "Eulerian", "Lagrangian", "fluid", "mesh", "distortion", "Abaqus"]
    },

    {
        "id": "Q15_TECHNICAL",
        "category": "technical_terminology",
        "expected_docs": ["017. TDR LTPP"],
        "question": "What is Time Domain Reflectometry (TDR) and how does it measure soil moisture content in pavement systems? Explain the relationship between dielectric constant and volumetric water content.",
        "difficulty": "hard",
        "answer_should_include": ["TDR", "electromagnetic", "dielectric constant", "volumetric moisture", "probe", "waveform"]
    },

    {
        "id": "Q16_TECHNICAL",
        "category": "technical_terminology",
        "expected_docs": ["012. Finite element"],
        "question": "Explain what Mean Texture Depth (MTD) is and how does it relate to runway friction performance, particularly regarding water drainage and hydroplaning prevention.",
        "difficulty": "medium",
        "answer_should_include": ["MTD", "texture", "water drainage", "hydroplaning", "friction", "surface morphology"]
    },

    {
        "id": "Q17_TECHNICAL",
        "category": "technical_terminology",
        "expected_docs": ["006. Rubber removal"],
        "question": "What is the difference between adhesional friction and hysteresis friction components, and how do they relate to pavement microtexture and macrotexture?",
        "difficulty": "hard",
        "answer_should_include": ["adhesional", "hysteresis", "microtexture", "macrotexture", "friction components"]
    },

    {
        "id": "Q18_TECHNICAL",
        "category": "technical_terminology",
        "expected_docs": ["014. SMA Specifications"],
        "question": "What is Stone Mastic Asphalt (SMA) and what are its key aggregate specifications and performance characteristics for airport pavements?",
        "difficulty": "medium",
        "answer_should_include": ["SMA", "aggregate", "specifications", "gap-graded", "stability"]
    },

    # ============================================================================
    # CATEGORÍA 5: PREGUNTAS DE COMPLEJIDAD OPERACIONAL
    # ============================================================================
    {
        "id": "Q19_OPERATIONAL",
        "category": "operational_complexity",
        "expected_docs": ["005. Snowfall impact"],
        "question": "Quantify the operational and economic impacts of snowfall on airport operations, including effects on delays, cancellations, and throughput capacity.",
        "difficulty": "medium",
        "answer_should_include": ["delays", "cancellations", "capacity", "economic", "operations", "snowfall"]
    },

    {
        "id": "Q20_OPERATIONAL",
        "category": "operational_complexity",
        "expected_docs": ["006. Rubber removal", "001. Winter friction"],
        "question": "What are the FAA or relevant aviation authority requirements and recommendations for runway friction levels, and how frequently should rubber removal be performed?",
        "difficulty": "medium",
        "answer_should_include": ["FAA", "friction", "requirements", "maintenance", "rubber removal", "frequency"]
    },

    # ============================================================================
    # CATEGORÍA 6: PREGUNTAS DE DISEÑO Y CONSTRUCCIÓN
    # ============================================================================
    {
        "id": "Q21_DESIGN",
        "category": "design_construction",
        "expected_docs": ["015. Carbon fiber grille"],
        "question": "Describe the construction details of the carbon fiber grille snow melting system, including the burial depth, spacing of heating wires, and integration with the cement concrete pavement structure.",
        "difficulty": "medium",
        "answer_should_include": ["5 cm depth", "10 cm spacing", "steel mesh", "48k carbon fiber", "concrete pavement"]
    },

    {
        "id": "Q22_DESIGN",
        "category": "design_construction",
        "expected_docs": ["006. Conductive asphalt"],
        "question": "What are the structural design considerations and lifecycle assessment findings for heated pavement systems using conductive asphalt on airport runways?",
        "difficulty": "hard",
        "answer_should_include": ["structural design", "lifecycle", "cost", "conductive asphalt", "heating system"]
    },

    # ============================================================================
    # CATEGORÍA 7: PREGUNTAS METODOLÓGICAS
    # ============================================================================
    {
        "id": "Q23_METHODS",
        "category": "methodology",
        "expected_docs": ["012. Finite element"],
        "question": "Describe the methodology used to create finite element meshes of asphalt runway surfaces, including the imaging techniques and mesh generation process.",
        "difficulty": "hard",
        "answer_should_include": ["X-ray", "laser profilometer", "CT scan", "mesh generation", "Simpleware", "texture"]
    },

    {
        "id": "Q24_METHODS",
        "category": "methodology",
        "expected_docs": ["006. Rubber removal"],
        "question": "What measurement protocols and statistical methods were used to evaluate and compare the performance of different rubber removal systems?",
        "difficulty": "medium",
        "answer_should_include": ["friction measurement", "texture measurement", "before/after", "statistical", "comparison"]
    },

    # ============================================================================
    # CATEGORÍA 8: PREGUNTAS FUERA DE ALCANCE (NO DEBERÍAN TENER RESPUESTA)
    # ============================================================================
    {
        "id": "Q25_OUT_OF_SCOPE",
        "category": "out_of_scope",
        "expected_docs": [],
        "question": "What are the best practices for aircraft engine maintenance and turbine blade inspection?",
        "difficulty": "n/a",
        "should_return": "no_relevant_documents",
        "answer_should_include": []
    },

    {
        "id": "Q26_OUT_OF_SCOPE",
        "category": "out_of_scope",
        "expected_docs": [],
        "question": "Explain the principles of machine learning algorithms used in computer vision systems.",
        "difficulty": "n/a",
        "should_return": "no_relevant_documents",
        "answer_should_include": []
    },

    {
        "id": "Q27_OUT_OF_SCOPE",
        "category": "out_of_scope",
        "expected_docs": [],
        "question": "What are the environmental regulations for airport noise pollution in European Union countries?",
        "difficulty": "n/a",
        "should_return": "no_relevant_documents",
        "answer_should_include": []
    },

    # ============================================================================
    # CATEGORÍA 9: PREGUNTAS AMBIGUAS (TESTING RETRIEVAL QUALITY)
    # ============================================================================
    {
        "id": "Q28_AMBIGUOUS",
        "category": "ambiguous",
        "expected_docs": ["Multiple possible"],
        "question": "How does temperature affect pavement performance?",
        "difficulty": "ambiguous",
        "note": "Could relate to heated pavements, seasonal effects, or material properties",
        "answer_should_include": ["temperature", "pavement"]
    },

    {
        "id": "Q29_AMBIGUOUS",
        "category": "ambiguous",
        "expected_docs": ["Multiple possible"],
        "question": "What sensors are used in pavement monitoring?",
        "difficulty": "ambiguous",
        "note": "Could relate to TDR sensors, thermometers, or other monitoring equipment",
        "answer_should_include": ["sensors", "monitoring"]
    },

    # ============================================================================
    # CATEGORÍA 10: PREGUNTAS DE RAZONAMIENTO COMPLEJO
    # ============================================================================
    {
        "id": "Q30_COMPLEX",
        "category": "complex_reasoning",
        "expected_docs": ["Multiple"],
        "question": "Considering the trade-offs between initial cost, operational efficiency, environmental impact, and long-term maintenance, what would be the most cost-effective solution for winter runway management at a high-traffic international airport in a region with moderate snowfall (10-20 snow days per year)?",
        "difficulty": "very_hard",
        "requires_integration": True,
        "answer_should_include": ["cost", "efficiency", "environment", "maintenance", "heated pavement", "chemicals", "trade-offs"]
    },
]

# Organizar por categoría para análisis
categories = {}
for q in test_questions:
    cat = q["category"]
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(q)

print(f"Total preguntas: {len(test_questions)}")
print(f"\nPreguntas por categoría:")
for cat, questions in categories.items():
    print(f"  {cat}: {len(questions)} preguntas")

