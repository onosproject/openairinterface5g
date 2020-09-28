Obtaining O-RAN WG3 E2 Specifications
=====================================

Before you can build the OAI O-RAN RIC agent, you will need to accept
the O-RAN Alliance Adopter License: https://www.o-ran.org/specifications .

Once you have an account, download all specifications in `.docx` form
from WG3: E2AP, E2SM-NI, and E2SM-KPM.  Place them in a temporary
directory on a machine that has the `openairinterface5g` repository.

For each .docx file, open and save as a plain .txt file with the same name.

Then run these commands:

openairinterface5g/cmake_targets/tools/extract_asn1_from_spec.pl \
    < ORAN-WG3.E2AP-v01.00.txt > ORAN-WG3.E2AP-v01.00.asn
openairinterface5g/cmake_targets/tools/extract_asn1_from_spec.pl \
    < ORAN-WG3.E2SM-NI-v01.00.txt > ORAN-WG3.E2SM-NI-v01.00.asn
openairinterface5g/cmake_targets/tools/extract_asn1_from_spec.pl \
    < ORAN-WG3.E2SM-KPM-v01.00.txt > ORAN-WG3.E2SM-KPM-v01.00.asn

This is version 01, so copy the .asn files into your openairinterface5g
tree:

mkdir -p openairinterface5g/openair2/RIC_AGENT/MESSAGES/ASN1/R01
cp ORAN-WG3.E2AP-v01.00.asn \
    ORAN-WG3.E2SM-NI-v01.00.asn \
    ORAN-WG3.E2SM-KPM-v01.00.asn \
    openairinterface5g/openair2/RIC_AGENT/MESSAGES/ASN1/R01

At this point, the build will work on these .asn files and generate the
necessary stubs via the `asn1c` compiler.
