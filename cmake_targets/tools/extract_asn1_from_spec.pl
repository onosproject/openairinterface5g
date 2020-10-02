#!/usr/bin/perl
# This script extracts the ASN1 definition from a 3GPP specification.
# First download the specification from 3gpp.org as a word document and open it
# Then in "view" menu, select normal or web layout (needed to removed page header and footers)
# Finally save the document as a text file
# Call the script: "perl extract_asn1_from_spec.pl <inputfile> <outputfile>"
# It will dump all ASN1 content blocks (between -- ASN1START and
# -- ASN1STOP, inclusive) to <outputfile>.

use strict;
use warnings;

my $alltofirst = 0;
if (@ARGV < 1) {
  $alltofirst = 1;
}
elsif (@ARGV > 1 && $ARGV[0] eq "-a") {
  $alltofirst = 1;
  shift;
}

my @output_files = ();
if (@ARGV > 0) {
  open(INPUT_FILE, "< $ARGV[0]") or die "Can not open file $ARGV[0]";
  if (@ARGV > 1) {
    for (my $i = 1; $i < @ARGV; ++$i) {
      my $fh;
      open($fh, "> $ARGV[$i]") or die "Can not open file $ARGV[$i]";
      push(@output_files,$fh);
    }
  }
  else {
    my $fh;
    open($fh, "> $ARGV[0].asn") or die "Can not open file $ARGV[0].asn";
    push(@output_files,$fh);
  }
}
else {
  *INPUT_FILE = *STDIN;
  push(@output_files,*STDOUT);
}

sub extract_asn1($);

if ($alltofirst) {
  while (extract_asn1($output_files[0])) {};
}
else {
  my $res = 1;
  for (my $i = 0; $i < @output_files; ++$i) {
    if ($res) {
	$res = extract_asn1($output_files[$i]);
    }
    close($output_files[$i]);
  }
}

close(INPUT_FILE);

# This subroutine copies the text delimited by -- ASN1START and -- ASN1STOP in INPUT_FILE
# and copies it into OUTPUT_FILE.
# It stops when it meets the keyword "END"
sub extract_asn1($) {
  my ($outfh,) = @_;
  my $line = <INPUT_FILE>;
  my $is_asn1 = 0;
  while(defined($line)) {
    if ($line =~ m/-- ASN1STOP/) {
      last;
    }
    if ($is_asn1 == 1) {
      syswrite($outfh,"$line");
    }
    if ($line =~ m/-- ASN1START/) {
      $is_asn1 = 1;
    }
    $line = <INPUT_FILE>;
  }
  return defined($line);
}
