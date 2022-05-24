/*
 * searchtools.js
 * ~~~~~~~~~~~~~~~~
 *
 * Sphinx JavaScript utilities for the full-text search.
 *
 * :copyright: Copyright 2007-2021 by the Sphinx team, see AUTHORS.
 * :license: BSD, see LICENSE for details.
 *
 */

if (!Scorer) {
  /**
   * Simple result scoring code.
   */
  var Scorer = {
    // Implement the following function to further tweak the score for each result
    // The function takes a result array [filename, title, anchor, descr, score]
    // and returns the new score.
    /*
    score: function(result) {
      return result[4];
    },
    */

    // query matches the full name of an object
    objNameMatch: 11,
    // or matches in the last dotted part of the object name
    objPartialMatch: 6,
    // Additive scores depending on the priority of the object
    objPrio: {0:  15,   // used to be importantResults
              1:  5,   // used to be objectResults
              2: -5},  // used to be unimportantResults
    //  Used when the priority is not in the mapping.
    objPrioDefault: 0,

    // query found in title
    title: 15,
    partialTitle: 7,
    // query found in terms
    term: 5,
    partialTerm: 2
  };
}

if (!splitQuery) {
  function splitQuery(query) {
    return query.split(/\s+/);
  }
}

/**
 * Search Module
 */
var KendraSearch = {

  _index : null,
  _queued_query : null,
  _pulse_status : -1,

  init : function() {
      var params = $.getQueryParameters();
      if (params.q) {
          var query = params.q[0];
          $('input[name="q"]')[0].value = query;
          this.performSearch(query);
      }
  },

  stopPulse : function() {
      this._pulse_status = 0;
  },

  startPulse : function() {
    if (this._pulse_status >= 0)
        return;
    function pulse() {
      var i;
      KendraSearch._pulse_status = (KendraSearch._pulse_status + 1) % 4;
      var dotString = '';
      for (i = 0; i < KendraSearch._pulse_status; i++)
        dotString += '.';
        KendraSearch.dots.text(dotString);
      if (KendraSearch._pulse_status > -1)
        window.setTimeout(pulse, 500);
    }
    pulse();
  },

  /**
   * perform a search for something (or wait until index is loaded)
   */
  performSearch : function(query) {
    console.log("Search Overriden");
    // create the required interface elements
    this.out = $('#search-results');
    this.title = $('<h2>' + _('Searching Overriden') + '</h2>').appendTo(this.out);
    this.dots = $('<span></span>').appendTo(this.title);
    this.status = $('<p class="search-summary">&nbsp;</p>').appendTo(this.out);
    this.output = $('<ul class="search"/>').appendTo(this.out);

    $('#search-progress').text(_('Preparing search...'));
    this.startPulse();

    this.query(query);
  },

  /**
   * execute search (requires search index to be loaded)
   */
  query : function(query) {
    var url = "https://nxnz7cg8n8.execute-api.us-west-2.amazonaws.com/dev/search_kendra";

    // array of [filename, title, anchor, descr, score]
    $('#search-progress').empty();

    console.log("query: ", query);

    fetch(url, {
      method: 'post',
      body: JSON.stringify({ "queryText": query }),
    }).then(response => response.json())
    .then(function(data) {
      console.log("data");
      console.log(data);
      var docs = JSON.parse(data.body)["ResultItems"];
      for(var i = 0; i < docs.length; i++){
          var listItem = $('<li></li>');
          var doc = docs[i];
          var doc_title = doc["DocumentTitle"]["Text"];
          var doc_url = doc["DocumentURI"];
          var text_excerpt = doc["DocumentExcerpt"]["Text"]
          var text_excerpt_highlights = doc["DocumentExcerpt"]["Highlights"]

          listItem.append($('<a/>').attr('href', doc_url).html(doc_title));
          
          resHTML = '';
          resHTML += text_excerpt.slice(0, text_excerpt_highlights[0]["BeginOffset"])
          for(var j = 0; j < text_excerpt_highlights.length; j++){
                resHTML += '<mark>';
                resHTML += text_excerpt.slice(text_excerpt_highlights[j]["BeginOffset"], text_excerpt_highlights[j]["EndOffset"]);

                resHTML += '</mark>';

                if( j + 1 == text_excerpt_highlights.length){
                  resHTML += text_excerpt.slice(text_excerpt_highlights[j]["EndOffset"]);
                }else{
                  resHTML += text_excerpt.slice(text_excerpt_highlights[j]["EndOffset"], text_excerpt_highlights[j+1]["BeginOffset"]);
                }
          }
          listItem.append($('<p/>').html(resHTML));

          // listItem.innerHTML = resHTML;
          KendraSearch.output.append(listItem);

      }
      
      
    }).catch(function(err) {
      console.log("err");
      console.log(err);
    });

    KendraSearch.stopPulse();
    KendraSearch.title.text(_('Search Results'));

    // for debugging
    //Search.lastresults = results.slice();  // a copy
    //console.info('search results:', Search.lastresults);

  },
  
  /**
   * See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions
   */
  escapeRegExp : function(string) {
    return string.replace(/[.*+\-?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
  }
};

$(document).ready(function() {
  KendraSearch.init();
});
